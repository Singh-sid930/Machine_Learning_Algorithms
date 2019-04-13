import numpy as np
import pandas as pd 
import string
from nltk.tokenize import RegexpTokenizer
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression




X_set = []
X_test = []

for i in range(1,24):

	for n in range(1,16):

		dirname  = os.path.dirname(__file__)
		filename = os.path.join(dirname, 'articles/articles/train/author_'+str(i)+'/'+str(n)+'.txt')
		with open(filename) as myfile:
		  data = myfile.read()
		
		data = re.sub(r'[^\w\s]','',data)

		X_set.append(data)


p=0
y_train = np.zeros(345)
for i in range(1,24):
	for n in range (1,16):
		y_train[p] = i
		p = p+1


for i in range(1,24):

	for n in range(16,21):

		dirname1  = os.path.dirname(__file__)
		filename1 = os.path.join(dirname, 'articles/articles/test/author_'+str(i)+'/'+str(n)+'.txt')
		with open(filename1) as myfile1:
		  data = myfile1.read()
		
		data = re.sub(r'[^\w\s]','',data)

		X_set.append(data)
		X_test.append(data)
		


p = 0
y_test = np.zeros(115)
for i in range(1,24):
	for n in range (16,21):
		y_test[p] = i
		p = p+1





vectorizer = CountVectorizer(stop_words='english',max_features=8000,ngram_range=(1,2))
X_feat = vectorizer.fit_transform(X_set)
feature_arr = X_feat.toarray()
feature_arr = np.array(feature_arr)
word_list = vectorizer.get_feature_names()



#**************************************Finding the top ten most frequently used words for the first 10 authors ***************************#


max_freq_train=[]
max_word_freq_train = []
i= 0
for p in range(10):
	word_freq = np.sum(feature_arr[i:i+15], axis=0)
	max_fr_ls = np.argpartition(word_freq, -10)[-10:]
	word_ls = []
	freq_ls = []
	for q in range(len(max_fr_ls)):
		word = word_list[max_fr_ls[q]]
		freq = word_freq[max_fr_ls[q]]
		word_ls.append(word)
		freq_ls.append(freq)

	max_freq_train.append(freq_ls)
	max_word_freq_train.append(word_ls)
	i=i+15

train_freq = np.array([max_word_freq_train,max_freq_train])


# print("*********************************************")

max_freq_test=[]
max_word_freq_test = []

i = 345
for p in range(10):
	word_freq = np.sum(feature_arr[i:i+5], axis=0)
	max_fr_ls = np.argpartition(word_freq, -10)[-10:]
	word_ls = []
	freq_ls = []
	for r in range(len(max_fr_ls)):
		word = word_list[max_fr_ls[r]]
		freq = word_freq[max_fr_ls[r]]
		word_ls.append(word)
		freq_ls.append(freq)

	max_freq_test.append(freq_ls)
	max_word_freq_test.append(word_ls)
	i=i+5

test_freq = np.array([max_word_freq_test,max_freq_test])

for i in range(10):
	print("******************************")
	print("author "+str(i+1))
	print(train_freq[0,i])
	print(test_freq[0,i])
	print(train_freq[1,i])
	print(test_freq[1,i])
	print("******************************")

# 



#********************************** Fitting model using Vectorizer and Multinomial Naive Bayes ******************************#



# feature_train_arr = feature_arr[0:345,:]
# feature_test_arr = feature_arr[345:460,:]
# # print(vectorizer.get_feature_names())
# # print(feature_arr.shape)
# clfmnb = MultinomialNB(alpha=0.02)
# clfmnb.fit(feature_train_arr, y_train)

# y_pred_train = clfmnb.predict(feature_train_arr)
# y_pred_test = clfmnb.predict(feature_test_arr)

# accuracy_train = accuracy_score(y_train,y_pred_train)
# accuracy_test = accuracy_score(y_test,y_pred_test)

# # print(accuracy_train)
# # print(accuracy_test)



# #*************************************** fitting model using TFLDF and different training models. *******************************#


# TF_IDFvectorizer = TfidfVectorizer(stop_words='english',max_features=8000,ngram_range=(1,2))
# X_feat= TF_IDFvectorizer.fit_transform(X_set)
# feature_arr = X_feat.toarray()
# feature_arr = np.array(feature_arr)

# feature_train_arr = feature_arr[0:345,:]
# word_list = TF_IDFvectorizer.get_feature_names()

# clfsvmrbf = SVC(kernel='rbf', C=0.1)
# clfsvmrbf.fit(feature_train_arr,y_train)

# y_pred_svm_rbf_train = clfsvmrbf.predict(feature_train_arr)
# y_pred_svm_rbf_test = clfsvmrbf.predict(feature_test_arr)

# accuracy_train = accuracy_score(y_train,y_pred_svm_rbf_train)
# accuracy_test = accuracy_score(y_test,y_pred_svm_rbf_test)


# # print(accuracy_train)
# # print(accuracy_test)


# clfsvml = SVC(kernel='linear', C=1.0)
# clfsvml.fit(feature_train_arr,y_train)


# y_pred_svm_lin_train = clfsvml.predict(feature_train_arr)
# y_pred_svm_lin_test = clfsvml.predict(feature_test_arr)

# accuracy_train = accuracy_score(y_train,y_pred_svm_lin_train)
# accuracy_test = accuracy_score(y_test,y_pred_svm_lin_test)

# # print(accuracy_train)
# # print(accuracy_test)


# X_norm = normalize(feature_train_arr, norm='l2', axis=1, copy=True, return_norm=False)
# clfsvmcos = SVC(kernel='linear', C=1.0)
# clfsvmcos.fit(X_norm,y_train)

# y_pred_svm_cos_train = clfsvmcos.predict(feature_train_arr)
# y_pred_svm_cos_test = clfsvmcos.predict(feature_test_arr)

# accuracy_train = accuracy_score(y_train,y_pred_svm_cos_train)
# accuracy_test = accuracy_score(y_test,y_pred_svm_cos_test)

# # print(accuracy_train)
# # print(accuracy_test)



# clfgnb = GaussianNB()
# clfgnb.fit(feature_train_arr,y_train)

# y_pred_gnb_train = clfgnb.predict(feature_train_arr)
# y_pred_gnb_test = clfgnb.predict(feature_test_arr)

# accuracy_train = accuracy_score(y_train,y_pred_gnb_train)
# accuracy_test = accuracy_score(y_test,y_pred_gnb_test)

# # print(accuracy_train)
# # print(accuracy_test)


# clfLR = LogisticRegression(random_state = 42, multi_class = 'multinomial', solver ='lbfgs')
# clfLR.fit(feature_train_arr,y_train)

# y_pred_LR_train = clfLR.predict(feature_train_arr)
# y_pred_LR_test = clfLR.predict(feature_test_arr)

# accuracy_train = accuracy_score(y_train,y_pred_LR_train)
# accuracy_test = accuracy_score(y_test,y_pred_LR_test)

# # print(accuracy_train)
# # print(accuracy_test)


