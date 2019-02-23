# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:35:00 2019

@author: Rutuja Moharil
"""

import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from logreg import LogisticRegression
from sklearn import svm, datasets
from logreg_adagrad import LogisticRegressionAdagrad
data=pd.read_csv('diabetes.dat',sep=',' , header= None)
x_dia=data.iloc[:,:-1].values
scores=[]
scores1=[]
scores2=[]
#x0=pd.DataFrame(x0)
y_dia=data.iloc[:,-1].values
for n in range(len(y_dia)):
    if y_dia[n]=='tested_positive':
        y_dia[n]=1
    elif y_dia[n]=='tested_negative':
        y_dia[n]=0

for i in range(5):
    
    x_train,x_test,y_train,y_test = train_test_split( x_dia, y_dia, test_size=0.2,shuffle = True )
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    x_train = (x_train- mean) / std
    x_test = (x_test - mean)/std
    # clf = LogisticRegressionAdagrad(alpha = 0.01, regLambda=0.001,regNorm=1)
    clf = svm.SVC(C=0.01,kernel='linear')
    clf1 = svm.SVC(C=1,kernel='linear')
    clf2 = svm.SVC(C=100,kernel='linear')
    y_train=np.array(y_train,dtype="float64")
 
    y_test=np.array(y_test,dtype="float64")
#    y_train=y_train.reshape(y_train.shape[0],1)
#    y_test=y_test.reshape(y_test.shape[0],1)
    clf.fit(x_train,y_train)
    clf1.fit(x_train,y_train)
    clf2.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    y_pred1 = clf1.predict(x_test)
    y_pred2 = clf2.predict(x_test)
    accuracy = accuracy_score(y_test,y_pred)
    accuracy1 = accuracy_score(y_test,y_pred1)
    accuracy2 = accuracy_score(y_test,y_pred2)
    scores.append(accuracy)
    scores1.append(accuracy1)
    scores2.append(accuracy2)
print("the score through SVM for the diabetes data set with C = 0.01 is = ",np.mean(scores))
print("the score through SVM for the diabetes data set with C = 1 is = ",np.mean(scores1))
print("the score through SVM for the diabetes data set with C = 100 is = ",np.mean(scores2))