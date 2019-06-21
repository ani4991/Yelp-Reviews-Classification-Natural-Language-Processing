# LIBRARIES IMPORT

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# IMPORT DATASET

yelp_df = pd.read_csv("yelp.csv")
print(yelp_df.head(10))
print(yelp_df.describe())

# VISUALIZE DATASET
# get the length of the messages
yelp_df['length'] = yelp_df['text'].apply(len)
print(yelp_df.head())

yelp_df['length'].plot(bins=100, kind='hist')

print(yelp_df.length.describe())

# longest message 43952
yelp_df[yelp_df['length'] == 4997]['text'].iloc[0]

# shortest message
yelp_df[yelp_df['length'] == 1]['text'].iloc[0]

# message with mean length
yelp_df[yelp_df['length'] == 710]['text'].iloc[0]

sns.countplot(y = 'stars', data=yelp_df)
g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=3)
g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5)
g.map(plt.hist, 'length', bins = 20, color = 'r')

# divide the reviews into 1 and 5 stars

yelp_df_1 = yelp_df[yelp_df['stars']==1]
yelp_df_5 = yelp_df[yelp_df['stars']==5]

yelp_df_1_5 = pd.concat([yelp_df_1 , yelp_df_5])
print(yelp_df_1_5.info())
print( '1-Stars percentage =', (len(yelp_df_1) / len(yelp_df_1_5) )*100,"%")

sns.countplot(yelp_df_1_5['stars'], label = "Count")

# CREATED TESTING AND TRAINING DATASET/DATA CLEANING
import string
string.punctuation
Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'
Test_punc_removed = [char for char in Test if char not in string.punctuation]
print(Test)

# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join

# REMOVE STOPWORDS

# download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')

print(Test_punc_removed_join)
Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
Test_punc_removed_join_clean # Only important (no so common) words are left

# COUNT VECTORIZER EXAMPLE
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)
print(vectorizer.get_feature_names())
print(X.toarray())

# define a pipeline to clean up all the messages
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
   Test_punc_removed = [char for char in message if char not in string.punctuation]
   Test_punc_removed_join = ''.join(Test_punc_removed)
   Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
   return Test_punc_removed_join_clean

# test the newly added function
yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)

print(yelp_df_clean[0]) # show the cleaned up version

from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])

# TRAINING THE MODEL WITH ALL DATASET

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values
NB_classifier.fit(yelp_countvectorizer, label)

testing_sample = ['amazing food! highly recommmended']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print(test_predict)

testing_sample = ['shit food, made me sick']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
print(test_predict)

# DIVIDEDTHE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING

X = yelp_countvectorizer
y = label
print(X.shape,y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

# EVALUATING THE MODEL

from sklearn.metrics import classification_report, confusion_matrix

y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)

print(classification_report(y_test, y_predict_test))

# ADDITION OF FEATURE TF-IDF

from sklearn.feature_extraction.text import TfidfTransformer

yelp_tfidf = TfidfTransformer().fit_transform(yelp_countvectorizer)
print(yelp_tfidf.shape)
print(yelp_tfidf[:,:])
# Sparse matrix with all the values of IF-IDF

X = yelp_tfidf
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)




















