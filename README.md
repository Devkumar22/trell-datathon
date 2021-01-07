# trell-datathon
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
trell_data = pd.read_csv("train_age_dataset.csv") 
trell_data


trell_data=pd.DataFrame(trell_data, columns=['tier','gender','following_rate','following_avg_age','followers_avg_age','max_repetitive_punc','num_of_hashtags_per_action','emoji_count_per_action','punctuations_per_action','number_of_words_per_action','avgCompletion','avgTimeSpent','avgDuration','creations','content_views','num_of_comments','weekends_trails_watched_per_day','weekdays_trails_watched_per_day','avgt2','age_group'])
trell_data

sns.barplot('age_group','followers_avg_age',data=trell_data)

sns.barplot('age_group','following_avg_age',data=trell_data)

feature_data=trell_data[[
        'tier',
        'gender',
        'following_rate',
        'following_avg_age',
        'followers_avg_age',
        'max_repetitive_punc',	
        'num_of_hashtags_per_action',
        'emoji_count_per_action',
        'punctuations_per_action',
        'number_of_words_per_action',
        'avgCompletion',
        'avgTimeSpent',	
        'avgDuration',
        'creations',
        'content_views',
        'num_of_comments',
        'weekends_trails_watched_per_day',
        'weekdays_trails_watched_per_day',
        'avgt2']].values
label_data=trell_data[['age_group']]  
      

stratified_fdata = stratified_sample(feature_data, strata=['tier',
        'gender',
        'following_rate',
        'following_avg_age',
        'followers_avg_age',
        'max_repetitive_punc',	
        'num_of_hashtags_per_action',
        'emoji_count_per_action',
        'punctuations_per_action',
        'number_of_words_per_action',
        'avgCompletion',
        'avgTimeSpent',	
        'avgDuration',
        'creations',
        'content_views',
        'num_of_comments',
        'weekends_trails_watched_per_day',
        'weekdays_trails_watched_per_day',
        'avgt2'], size=0.3)
stratified_ldata= stratified_sample(label_data, strata=['age_group'],test_size=0.3)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(feature_data,label_data,test_size=0.4)

#KNN
from sklearn import neighbors,metrics

knn=neighbors.KNeighborsClassifier(n_neighbors=699, weights='distance', algorithm='auto')

knn.fit(X_train,y_train)

target=knn.predict(X_test)

accuracy= metrics.accuracy_score(y_test,target)

print("predic:", target)
print("accuracy-score:", accuracy)

#print('actualvalue', y[2])
#print('prediction:', knn.predict(X)[172])

#RANDOM-FOREST
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, y_train)
clf.predict(X_test)
accuracy= metrics.accuracy_score(y_test[0:54320],rf_predicted_values)
print("accuracy:",accuracy)

#naive-bayes
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
nbayes = MultinomialNB()
nbayes.fit(X_train,y_train)


predict = nbayes.predict(X_test)
accuracy= metrics.accuracy_score(y_test,predict)
print('pre', predict)
print('acc', accuracy)

from sklearn.externals import joblib             
nb_save = ('nbayes.sav')
joblib.dump(nbayes, nb_save)

test_data= pd.read_csv("test_age_dataset.csv")
nbayes = joblib.load(nb_save)
answer = pd.DataFrame(test_data, columns=['tier','gender','following_rate','following_avg_age','followers_avg_age','max_repetitive_punc','num_of_hashtags_per_action','emoji_count_per_action','punctuations_per_action','number_of_words_per_action','avgCompletion','avgTimeSpent','avgDuration','creations','content_views','num_of_comments','weekends_trails_watched_per_day','weekdays_trails_watched_per_day','avgt2'])
answer.head()

nb_predicted_values = nbayes.predict(answer)
print("prediction:", nb_predicted_values)
accuracy= metrics.accuracy_score(predict[0:54320],nb_predicted_values)
print('acccc:', accuracy)

from sklearn.metrics import f1_score
f1_score(predict[0:54320], nb_predicted_values,average='weighted')

res=pd.DataFrame(nb_predicted_values)
res.index=test_data.index
res.columns=['prediction']

from google.colab import files 
res.to_csv('nb_prediction_result.csv')
files.download('nb_prediction_result.csv')

sns.heatmap(feature_data)
