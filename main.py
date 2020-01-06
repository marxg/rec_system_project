import pandas as pd
import seaborn as sns
import matplotlib as plt
import numpy as np
import ast
import pickle
import xgboost as xgb
import gc
import random

from average_precision import apk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from random import shuffle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from scipy.sparse import coo_matrix

def process_user_kw(components=20):    
    user_kw=pd.read_csv('user_key_word.txt', sep='\t', names=['userid', 'keyword'])
    r=[]
    c=[]
    e=[]
    for entry in user_kw.iterrows():
        for pair in entry[1][1].split(';'):
            kw,weight=pair.split(':')
            weight=float(weight)
            kw=int(kw)
            if weight == 2:
                continue
            r.append(entry[0])
            c.append(kw)
            e.append(weight)
    matrix=coo_matrix((e, (r, c)))
    print(matrix.shape)
    svd=TruncatedSVD(components)
    X_svd=svd.fit_transform(matrix)
    print(svd.explained_variance_ratio_.sum())
    user_kw=user_kw.join(pd.DataFrame(X_svd, columns=[f'user_kw_{n}' for n in range(components)]))
    user_kw=user_kw.drop(columns=['keyword'])
    return user_kw

def process_log_data(log_data, size):  
    log_data = log_data.drop_duplicates(subset=['userid','itemid'])
    log_data = log_data.dropna()
    log_data = log_data.sample(frac=size)
    log_data['time']=pd.to_datetime(log_data['unix_time'], unit='s')
    log_data['time']=log_data['time'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    log_data['day']=log_data['time'].apply(lambda x : x.dayofweek)
    log_data['hour']=log_data['time'].apply(lambda x : x.hour)
#    log_data['month']=log_data['time'].apply(lambda x : x.month)
#    log_data = log_data.join(pd.get_dummies(log_data.day, prefix='day'))
#    log_data = log_data.join(pd.get_dummies(log_data.hour, prefix='hour'))
#    log_data = log_data.join(pd.get_dummies(log_data.month, prefix='month'))
    log_data= log_data.drop(columns=['time','day','hour'])
    return log_data

def process_user_profiles(components=20):    
    user_profiles = pd.read_csv('user_profile.txt', sep='\t', names=['userid', 'yob', 'gender', 'num_tweets', 'tagid' ])
    user_profiles['gender']=[0 if x==3 else x for x in user_profiles.gender]
    user_profiles['yob']=[x if x.isdigit() else 1990 for x in user_profiles.yob]
    user_profiles['yob']=pd.to_numeric(user_profiles.yob)
    user_profiles = user_profiles.join(pd.get_dummies(user_profiles.gender, prefix='gender'))
    user_profiles = user_profiles.drop(columns = ['gender'])
    text=user_profiles.tagid.apply(lambda x : x.replace(';',' '))
    vectorizer = CountVectorizer()
    X=vectorizer.fit_transform(text)
    print(X.shape)
    svd=TruncatedSVD(n_components=components)
    X_svd=svd.fit_transform(X)
    print(svd.explained_variance_ratio_.sum())
    user_profiles=user_profiles.join(pd.DataFrame(X_svd, columns=[f'user_tag_{n}' for n in range(components)]))
    user_profiles=user_profiles.drop(columns=['tagid'])
    return user_profiles

def prep_train_test(train_size=.1, test_size=1, user_tag_dim=20, user_kw_dim= 20, item_dim=20):
    #train data
    rec_log_col_names=['userid', 'itemid', 'result', 'unix_time']
    log_data_train = pd.read_csv('rec_log_train.txt', sep="\t", names=rec_log_col_names)
    log_data_train = process_log_data(log_data_train, train_size)
    #test data
    with open('test.pickle', 'rb') as f:
        log_data_test = pickle.load(f)
    log_data_test = process_log_data(log_data_test, test_size)
    #item data - add features
    item_data = process_item_data(item_dim)
    #user data
    user_kw = process_user_kw(user_kw_dim)
    user_data = process_user_profiles(user_tag_dim)
    user_data = user_data.merge(user_kw, how='left', on ='userid')
    #recheck that not lots of na below
    train=log_data_train.merge(item_data, how='left', on='itemid').merge(user_data, how='left', on='userid').fillna(value=0)
    test=log_data_test.merge(item_data, how='left', on='itemid').merge(user_data, how='left', on='userid').fillna(value=0)
    return (train, test)

def process_item_data(item_dim=20):
    with open('item_data.pickle', 'rb') as f:
        item_data = pickle.load(f)
    X=item_data.loc[:,'cat_0_1':]
    svd=TruncatedSVD(10)
    X_svd=svd.fit_transform(X)
    print(svd.explained_variance_ratio_.sum())
    item_data=item_data.loc[:,:'mean'].join(pd.DataFrame(X_svd, columns=[f'item_cat_{n}' for n in range(10)]))
    text=item_data.item_keyword.apply(lambda x : x.replace(';',' '))
    vectorizer = CountVectorizer()
    X=vectorizer.fit_transform(text)
    print(X.shape)
    svd=TruncatedSVD(item_dim)
    X_svd=svd.fit_transform(X)
    print(svd.explained_variance_ratio_.sum())
    item_data=item_data.join(pd.DataFrame(X_svd, columns=[f'item_kw_{n}' for n in range(item_dim)]))
    item_data=item_data.drop(columns=['item_keyword','item_category'])
    return item_data

def get_xgb_imp(xgb, feat_names):
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = np.array(imp_dict.values()).sum()
    return {k:v/total for k,v in imp_dict.items()}

def find_top_three(user_item_scores):
    gb=user_item_scores.groupby('userid')
    top_three={}
    for group in gb:
        group_sorted=group[1].sort_values('scores', ascending=False)
        top_three[group[1].userid.iloc[0]]=list(group_sorted.itemid)[:3]
    return top_three           

def one_two_three(model, train, test, *args, **kwargs):
    X_train=train.drop(columns=['result','userid','itemid'])
    X_test=test.drop(columns=['result','userid','itemid'])
    ###Changing to 0 and 1
    y_train=[0 if x==-1 else x for x in train.result]
    y_test=[]
    for (x,y) in zip(test.userid, test.itemid):
        result=0
        if x in solution_dict:
            if y in solution_dict[x]:
                result=1
        y_test.append(result)
    try:
        name = model.__name__
    except:
        name='xgb'
    mm=model(*args, **kwargs)
    mm.fit(X_train, y_train)
    y_predict=mm.predict(X_test)
    y_predict_proba=mm.predict_proba(X_test)
    ####################################
    acc=accuracy_score(y_test, y_predict)
    prec=precision_score(y_test, y_predict)
    f1=f1_score(y_test, y_predict)
    recall=recall_score(y_test, y_predict)
    print("Train-test split:")
    print(f"{name} accuracy is {acc}")
    print(f"{name} precision is {prec}")
    print(f"{name} f1 is {f1}")
    print(f"{name} recall is {recall}")
    roc_auc=roc_auc_score(y_test, y_predict_proba[:,1])
    print(f"{name} roc auc is {roc_auc}")
    roc_auc_train=roc_auc_score(y_train, mm.predict_proba(X_train)[:,1])
    print(f"{name} train roc auc is {roc_auc_train}")
    #######################################
    user_item_scores=test.loc[:,['userid','itemid']]
    user_item_scores.loc[:, 'scores']=y_predict_proba[:,1]
    top_three_pre=find_top_three(user_item_scores)
    map_score=[]
    for x in top_three_pre:
        if x in solution_dict:
            map_score.append(apk(solution_dict[x],top_three_pre[x]))
    map_score=np.mean(map_score)
    print(f"{name} mean avg precision in {map_score}")
    if name == "XGBClassifier":
        xgb.plot_importance(mm)
        plt.title("xgboost.plot_importance(model)")
        plt.show()
        print(get_xgb_imp(mm, train.columns))
    if name == "RandomForestClassifier":
        fi=list(zip(X_train.columns,mm.feature_importances_))
        fi.sort(key = lambda x : x[1], reverse=True)
        print(fi)

if __name__ == "__main__":    
    with open('solution_dict.pickle', 'rb') as f:
        solution_dict = pickle.load(f)

    (train,test)=prep_train_test(.3,1,30,30,30)

    one_two_three(RandomForestClassifier, train, test,
                n_estimators = 100, 
                min_impurity_decrease=.00005,
                max_features="auto",
                class_weight="balanced",
                n_jobs=4
                )
