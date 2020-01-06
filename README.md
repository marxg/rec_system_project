# Readme
## Recommendation System for Tencent Weibo (Twitter Clone)

### Description: 
I built a content recommendation system for a Twitter-like social media platform created by Tencent. The project was an exercise in developing content for new users - the so-called cold start problem. We have some basic information about new users, i.e., User 11 is interested in mountain biking, swimming, and kpop, but we don't know what content **in our system** will appeal to the user most.
A simple (but pretty good) approach to the problem is to recommend the most popular content on the system, ignoring the user-specific data altogether. To do better, we extracted meaning from the text-based data that we obtained from the user at registration using natural language processing and dimension reduction techniques.

### File Directory:
- main.py: Contains scripts for loading data, processing, feature engineering, modeling, and evaluation
- slides.pdf: Presentation slides prepared for Metis ~ 12/2019
