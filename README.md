# Readme
## Recommendation System for Tencent Weibo (Twitter Clone)

### Description: 
I built a content recommendation system for **new users** of a Twitter-like social media platform created by Tencent. This is an example of a "cold start problem". We have some basic information about the user, i.e., Jake123 is interested in mountain biking, swimming, and k-pop, but we don't know what content in our system will appeal most to the user. A simple (but pretty good) approach to the problem is to recommend the most popular content on the system, ignoring the user-specific data altogether. To do better, we used natural language processing and dimension reduction techniques to incorporate users' profile data into our model.

### File Directory:
- main.py: Contains scripts for loading data, processing, feature engineering, modeling, and evaluation
- slides.pdf: Presentation slides prepared for Metis ~ 12/2019
