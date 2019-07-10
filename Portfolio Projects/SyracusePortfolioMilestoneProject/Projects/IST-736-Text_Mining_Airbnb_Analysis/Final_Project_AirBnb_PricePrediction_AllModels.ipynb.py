#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 16:52:56 2019

@author: GiorgiJ
"""
# -------------------------------------- LIBRARY -------------------------------------- # 

import nltk
import pandas as pd
import sklearn
import re  
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import numpy as np
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import string
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize          
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

# %% -------------------------------------- READ IN FILE -------------------------------------- # 

Init_file = "/Users/GiorgiJ/Desktop/Syracuse University/IST 736/Project/DATA/reviews.csv"
Init_file = pd.read_csv(Init_file)
drop_col = ['listing_id', 'id', 'date', 'reviewer_id', 'reviewer_name', 'Unnamed: 0']
#Init_file = Init_file.drop(drop_col, axis=1)

# %% -------------------------------------- CHANGE POLARITY INTO SENTIMENT -------------------------------------- # 

Init_file['polarity'] = pd.DataFrame(np.where(Init_file['polarity'] < 0, 'Negative', 'Positive'))
Init_file = Init_file.rename(columns = {'polarity':'Sentiment'})
Init_file = Init_file.sample(frac = 0.015, random_state = 13)
Init_file = Init_file.reset_index()
Init_file = Init_file.drop(Init_file.columns[0], axis = 1)

x = pd.DataFrame(Init_file['comments'])
Init_file = Init_file.drop(Init_file.columns[0], axis = 1)
Init_file = pd.concat([Init_file, x], axis = 1)

# %% -------------------------------------- CLEAN DATA -------------------------------------- # 

Init_file['comments'] = Init_file['comments'].apply(lambda x: " ".join(x.lower() for x in x.split()))
Init_file['comments'] = Init_file['comments'].str.replace('[^\w\s]','')
Init_file['comments'] = Init_file['comments'].str.replace('[0-9]','')
stop = stopwords.words('english')
Init_file['comments'] = Init_file['comments'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

y=Init_file['Sentiment'].values
X=Init_file['comments'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
# %% -------------------------------------- LEMMAZTIZER --------------------------------------- # 

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]

tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),
                                strip_accents = 'unicode',
                                stop_words = 'english',
                                lowercase = True,
                                max_df = 0.5,
                                min_df = 10) 

# %% -------------------------------------- VECTORIZERS --------------------------------------- # 

# Boolean Vectorizers
Bool = CountVectorizer(input="content", tokenizer = LemmaTokenizer(), encoding = 'latin-1', binary = True, 
                         analyzer = "word")

#Freq Vectororizers
Freq = CountVectorizer(input="content", tokenizer = LemmaTokenizer(), encoding = 'latin-1', binary = False, 
                         analyzer = "word")

# %% ------------------------------------------------------------------------------------------ #
Bool_Vect = Bool.fit_transform(X_train)
Column_Names_1 = Bool.get_feature_names()
B_Vector_DF = pd.DataFrame(Bool_Vect.toarray(),columns= Column_Names_1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
Freq_Vect = Freq.fit_transform(X_train)
Column_Names_2 = Freq.get_feature_names()
F_Vector_DF = pd.DataFrame(Freq_Vect.toarray(),columns= Column_Names_2)

# %% -------------------------------------- RE-LABEL DFs --------------------------------------- # 
# %% -------------------------------------- VECTORIZE TEST DATA -------------------------------------- # 

Bool_vec = Bool.transform(X_test)

Freq_vec = Freq.transform(X_test)

#######################################################################################################
############################################# NAIVE BAYES #############################################
#######################################################################################################

# %% -------------------------------------- MNB -------------------------------------- # 

MNB = MultinomialNB()
MNB.fit(Freq_Vect,y_train)

# %% -------------------------------------- MNB FEATURES -------------------------------------- # 

def show_most_and_least_informative_features(vectorizer,clf,class_idx=0, n=10): 
    feature_names = vectorizer.get_feature_names() 
    coefs_with_fns = sorted(zip(clf.coef_[class_idx], feature_names)) 
    top = zip(coefs_with_fns[:n], coefs_with_fns[-n:]) 
    for (coef_1, fn_1), (coef_2, fn_2) in top: 
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)) 

# %% ------------------------------------------------------------------------------------------ #        

show_most_and_least_informative_features(Freq, MNB, class_idx=0, n= 10) 

# %% -------------------------------------- TEST MNB -------------------------------------- # 
a = MNB.score(Freq_vec, y_test)

# %% ------------------------------------------------------------------------------------------ #    

y_pred = MNB.fit(Freq_Vect, y_train).predict(Freq_vec)
cm=confusion_matrix(y_test, y_pred, labels=['Negative','Positive'])
print(cm)

target_names = ['Negative','Positive']
print(classification_report(y_test, y_pred, target_names=target_names))


# %% -------------------------------------- BNB -------------------------------------- # 
BNB = BernoulliNB()
BNB.fit(Bool_Vect,y_train)

# %% -------------------------------------- BNB FEATURES -------------------------------------- # 

def show_most_and_least_informative_features(vectorizer,clf,class_idx=0, n=10): 
    feature_names = vectorizer.get_feature_names() 
    coefs_with_fns = sorted(zip(clf.coef_[class_idx], feature_names)) 
    top = zip(coefs_with_fns[:n], coefs_with_fns[-n:]) 
    for (coef_1, fn_1), (coef_2, fn_2) in top: 
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2)) 

# %% ------------------------------------------------------------------------------------------ #        

show_most_and_least_informative_features(Bool, BNB, class_idx=0, n= 10) 

# %% -------------------------------------- TEST MNB -------------------------------------- # 
b = BNB.score(Freq_vec, y_test)

# %% ------------------------------------------------------------------------------------------ #    

z_pred = BNB.fit(Bool_Vect, y_train).predict(Bool_vec)
cm=confusion_matrix(y_test, z_pred, labels=['Negative','Positive'])
print(cm)

target_names = ['Negative','Positive']
print(classification_report(y_test, z_pred, target_names=target_names))
 
 # %% - - - - - - - - - - - - - - - - - SENT BNB MODEL  - - - - - - - - - - - - - - - - - #

# %% -------------------------------------- SVC -------------------------------------- # 
SVM_1 = LinearSVC(C=1)
SVM_1.fit(Freq_Vect,y_train)

# %% -------------------------------------- SVM FEATURES -------------------------------------- # 
feature_ranks = sorted(zip(SVM_1.coef_[0], Freq.get_feature_names()))

## get the 10 features that are best indicators of very negative sentiment (they are at the bottom of the ranked list)
Negative_10 = feature_ranks[-10:]
print("Very negative words")
for i in range(0, len(Negative_10)):
    print(Negative_10[i])
print()

## get 10 features that are least relevant to "very negative" sentiment (they are at the top of the ranked list)
Positive_10 = feature_ranks[:10]
print("Not very negative words")
for i in range(0, len(Positive_10)):
    print(Positive_10[i])
print()

# %% -------------------------------------- TEST SVC -------------------------------------- # 
c = SVM_1.score(Freq_vec, y_test)

# %% ------------------------------------------------------------------------------------------ #    

w_pred = SVM_1.predict(Freq_vec)
cm=confusion_matrix(y_test,w_pred, labels=['Negative','Positive'])
print(cm)

target_names = ['Negative','Positive']
print(classification_report(y_test, w_pred, target_names=target_names))


# %% -------------------------------------- ALL RESULTS --------------------------------------- # 

Acc_List = [a,b,c]

Acc_Name = ['MNB', 'BNB', "SVC"]

Acc_DF = pd.DataFrame({'Model Name':Acc_Name, 'Accuracy %':Acc_List})
print(Acc_DF)












