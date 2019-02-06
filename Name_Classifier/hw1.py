#!/usr/bin/env python
# coding: utf-8

# # Homework 1 Template
# This is the template for the first homework assignment.
# Below are some function templates which we require you to fill out.
# These will be tested by the autograder, so it is important to not edit the function definitions.
# The functions have python docstrings which should indicate what the input and output arguments are.

# ## Instructions for the Autograder
# When you submit your code to the autograder on Gradescope, you will need to comment out any code which is not an import statement or contained within a function definition.

# In[2]:


# Uncomment and run this code if you want to verify your `sklearn` installation.
# If this cell outputs 'array([1])', then it's installed correctly.

# from sklearn import tree
# X = [[0, 0], [1, 1]]
# y = [0, 1]
# clf = tree.DecisionTreeClassifier(criterion='entropy')
# clf = clf.fit(X, y)
# clf.predict([[2, 2]])


# In[5]:


# Uncomment this code to see how to visualize a decision tree. This code should
# be commented out when you submit to the autograder.
# If this cell fails with
# an error related to `pydotplus`, try running `pip install pydotplus`
# from the command line, and retry. Similarly for any other package failure message.
# If you can't get this cell working, it's ok - this part is not required.
#
# This part should be commented out when you submit it to Gradescope

#from sklearn.externals.six import StringIO  
#from IPython.display import Image  
#import pydotplus
#
#dot_data = StringIO()
#tree.export_graphviz(clf, out_file=dot_data,  
#                filled=True, rounded=True,
#                special_characters=True,
#                feature_names=['feature1', 'feature2'],
#                class_names=['0', '1'])
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
#Image(graph.create_png())


# In[3]:


# This code should be commented out when you submit to the autograder.
# This cell will possibly download and unzip the dataset required for this assignment.
# It hasn't been tested on Windows, so it will not run if you are running on Windows.

#import os
#
#if os.name != 'nt':  # This is the Windows check
#    if not os.path.exists('badges.zip'):
#        # If your statement starts with "!", then the command is run in bash, not python
#        !wget https://www.seas.upenn.edu/~cis519/fall2018/assets/HW/HW1/badges.zip
#        !mkdir -p badges
#        !unzip badges.zip -d badges
#        print('The data has saved in the "badges" directory.')
#else:
#    print('Sorry, I think you are running on windows. '
#          'You will need to manually download the data')


# In[4]:

import string

def compute_features(name):

	"""
	Compute all of the features for a given name. The input
	name will always have 3 names separated by a space.
	
	Args:
		name (str): The input name, like "bill henry gates".
	Returns:
		list: The features for the name as a list, like [0, 0, 1, 0, 1].
	"""
	first, middle, last = name.split()

	name_features = []

	character_list  =   list(string.ascii_lowercase)

	if (len(first)<5):
		first = first + '#' * (5-len(first))

	if (len(middle)<5):
		middle = middle + '#' * (5-len(middle))

	if (len(last)<5):
		last = last + '#' * (5-len(last))

	full_name = first[:5] + middle[:5] + last[:5]

	for i in full_name:

		base_list = [0]*26

		if i in character_list:
			base_list[character_list.index(i)] = 1
			name_features.extend(base_list)
		else:
			name_features.extend(base_list)

	return(name_features)

# In[5]:


import numpy as np 

from sklearn.tree import DecisionTreeClassifier

# The `max_depth=None` construction is how you specify default arguments
# in python. By adding a default argument, you can call this method in a couple of ways:
#     
#     train_decision_tree(X, y)
#     train_decision_tree(X, y, 4) or train_decision_tree(X, y, max_depth=4)
#
# In the first way, max_depth is automatically set to `None`, otherwise it is 4.
def train_decision_tree(X, y, max_depth=None):
	"""
	Trains a decision tree on the input data using the information gain criterion
	(set the criterion in the constructor to 'entropy').
	
	Args:
		X (list of lists): The features, which is a list of length n, and
						   each item in the list is a list of length d. This
						   represents the n x d feature matrix.
		y (list): The n labels, one for each item in X.
		max_depth (int): The maximum depth the decision tree is allowed to be. If
						 `None`, then the depth is unbounded.
	Returns:
		DecisionTreeClassifier: the learned decision tree.
	"""
	classifier_1 = DecisionTreeClassifier(criterion = 'entropy', max_depth = max_depth)
	n_classifier = classifier_1.fit(np.array(X),y)
	return n_classifier


# In[6]:


from sklearn.linear_model import SGDClassifier

def train_sgd(X, y, learning_rate='optimal', alpha = 0.005, tol = 0.001):
	"""
	Trains an `SGDClassifier` using 'log' loss on the input data.
	
	Args:
		X (list of lists): The features, which is a list of length n, and
						   each item in the list is a list of length d. This
						   represents the n x d feature matrix.
		y (list): The n labels, one for each item in X.
		learning_rate (str): The learning rate to use. See http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
	Returns:
		SGDClassifier: the learned classifier.
	"""
	classifier_2 = SGDClassifier(loss = "log", alpha = alpha, tol = tol)
	s_classifier = classifier_2.fit(np.array(X),y)
	return s_classifier


# In[7]:

from sklearn.model_selection import train_test_split
def train_sgd_with_stumps(X, y):
	"""
	Trains an `SGDClassifier` using 'log' loss on the input data. The classifier will
	be trained on features that are computed using decision tree stumps.
	
	This function will return two items, the `SGDClassifier` and list of `DecisionTreeClassifier`s
	which were used to compute the new feature set. If `sgd` is the name of your `SGDClassifier`
	and `stumps` is the name of your list of `DecisionTreeClassifier`s, then writing
	`return sgd, stumps` will return both of them at the same time.
	
	Args:
		X (list of lists): The features, which is a list of length n, and
						   each item in the list is a list of length d. This
						   represents the n x d feature matrix.
		y (list): The n labels, one for each item in X.
	Returns:
		SGDClassifier: the learned classifier.
		List[DecisionTree]: the decision stumps that were used to compute the features
							for the `SGDClassifier`.
	"""
	# This is an example for how to return multiple arguments
	# in python. If you write `a, b = train_sgd_with_stumps(X, y)`, then
	# a will be 1 and b will be 2.

	stumps_feature  = []
	X_new_features  = []
	X1              = []

	for i in range(200):
		x_new       = []
		X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.5)
		dt_stump_classifier = DecisionTreeClassifier(max_depth= 8,criterion='entropy')
		dt_stump    = dt_stump_classifier.fit(X_train,y_train)
		stumps_feature.append(dt_stump)

		x_new       = dt_stump.predict(X)

		X1.append(x_new)

	X_new_features  = np.transpose(X1)

	sgd_classifier  = SGDClassifier(loss='log')
	sgd             = sgd_classifier.fit(np.array(X_new_features),y)


	return sgd,stumps_feature


# In[8]:


# The input to this function can be an `SGDClassifier` or a `DecisionTreeClassifier`.
# Because they both use the same interface for predicting labels, the code can be the same
# for both of them.
def predict_clf(clf, X):
	"""
	Predicts labels for all instances in `X` using the `clf` classifier. This function
	will be the same for `DecisionTreeClassifier`s and `SGDClassifier`s.
	
	Args:
		clf: (`SGDClassifier` or `DecisionTreeClassifier`): the trained classifier.
		X (list of lists): The features, which is a list of length n, and
						   each item in the list is a list of length d. This
						   represents the n x d feature matrix.
	Returns:
		List[int]: the predicted labels for each instance in `X`.
	"""
	return clf.predict(X)


# In[9]:


# The SGD-DT classifier can't use the same function as the SGD or decision trees
# because it requires an extra argument

def predict_sgd_with_stumps(sgd, stumps, X):
	"""
	Predicts labels for all instances `X` using the `SGDClassifier` trained with
	features computed from decision stumps. The input `X` will be a matrix of the
	original features. The stumps will be used to map `X` from the original features
	to the features that the `SGDClassifier` were trained with.
	
	Args:
		sgd (`SGDClassifier`): the classifier that was trained with features computed
							   using the input stumps.
		stumps (List[DecisionTreeClassifier]): a list of `DecisionTreeClassifier`s that
											   were used to train the `SGDClassifier`.
		X (list of lists): The features that were used to train the stumps (i.e. the original
						   feature set).
	Returns:
		List[int]: the predicted labels for each instance in `X`.
	"""
	X_new_features     =   []
	X1          =   []

	for stump in stumps:
		x_new   =   []
		x_new   =   stump.predict(X)
		X1.append(x_new)

	X_new_features = np.transpose(X1)

	return predict_clf(sgd,X_new_features)


# In[10]:


# Write the rest of your code here. Anything from here down should be commented
# out when you submit to the autograder


# In[ ]:

import os


############################################### Create Split Data Sets ###############################################

def data_set(fold):
	Y_set   =   []
	X_set   =   []
	name_list    =   []

	with open('train.fold-'+str(fold)+'.txt' , 'r') as f:
		for line in f:
			line_content = line.strip()
			content_word = line_content.split()
			if content_word[0] == '+':
				Y_set.append(1)
			else:
				Y_set.append(0)

			name_list.append(content_word[1]+ " " + content_word[2] + " " + content_word[3])
		   
		for name in name_list:
			feature_X =compute_features(name)
			X_set.append(feature_X)
	return(X_set,Y_set)



X0,y0 = data_set(0)
X1,y1 = data_set(1)
X2,y2 = data_set(2)
X3,y3 = data_set(3)
X4,y4 = data_set(4)

X_complete = [X0,X1,X2,X3,X4]
y_complete = [y0,y1,y2,y3,y3]

foldX_0 = X1+X2+X3+X4
foldy_0 = y1+y2+y3+y4
test0   = X0

foldX_1 = X0+X2+X3+X4
foldy_1 = y0+y2+y3+y4
test1   = X1

foldX_2 = X0+X1+X3+X4
foldy_2 = y0+y1+y3+y4
test2   = X2

foldX_3 = X0+X2+X1+X4
foldy_3 = y0+y2+y1+y4
test3   = X3

foldX_4 = X1+X2+X3+X0
foldy_4 = y1+y2+y3+y0
test4   = X4

# In[ ]:

folds_X_complete    = [foldX_0,foldX_1,foldX_2,foldX_3,foldX_4]
folds_y_complete    = [foldy_0,foldy_1,foldy_2,foldy_3,foldy_4]
X_test              = [X0,X1,X2,X3,X4]



from sklearn.metrics import accuracy_score
import pandas 
import csv


#####################################################Train and check SDG Classifier #################################################################

######################### SDG With varying alpha #######################

def sgd_alpha(alpha1):
	pa_list     = []
	ta_list     = []
	total_ta    = 0
	total_pa    = 0

	for i in range(5):

		sgd_class   =   train_sgd(folds_X_complete[i],folds_y_complete[i],alpha=alpha1)


		y_test      =   predict_clf(sgd_class,X_test[i])
		pa_score    =   accuracy_score(y_test,y_complete[i])
		pa_list.append(pa_score)

		y_train     =   predict_clf(sgd_class,folds_X_complete[i])
		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
		ta_list.append(ta_score)

		total_pa    =   total_pa + pa_score
		total_ta    =   total_ta + ta_score   

	pa = total_pa/5
	ta = total_ta/5

	return pa, ta, pa_list, ta_list


# print("printing SGD classifier values ")

pa_sgd_score, ta_sgd_score, pa_sgd, ta_sgd = sgd_alpha(0.001)

pa_sgd_list = pa_sgd


# alp1 = 0.001
# alp2 = 0.005


# for i in range(3):

# 	pa_sgd_score, ta_sgd_score, pa_sgd, ta_sgd = sgd_alpha(alp1)
# 	print("aplha = ")
# 	print (alp1)
# 	print("pa= ")
# 	print(pa_sgd_score)
# 	print("TA= ")
# 	print(ta_sgd_score)
	

# 	print("#######################SDG with varying Alpha print ############# ")

# 	pa_sgd_score, ta_sgd_score, pa_sgd, ta_sgd = sgd_alpha(alp2)

# 	print("aplha = ")
# 	print (alp2)
# 	print("pa= ")
# 	print(pa_sgd_score)
# 	print("TA= ")
# 	print(ta_sgd_score)

# 	print("#######################SDG with varying Alpha print ############# ")

	

# 	alp1 = alp1/10
# 	alp2 = alp2/10

# ################################# SDG with varying tolerance #############


# def sgd_tolerance(toler1):
# 	pa_list     = []
# 	ta_list     = []
# 	total_ta    = 0
# 	total_pa    = 0

# 	for i in range(5):
# 		sgd_class   =   train_sgd(folds_X_complete[i],folds_y_complete[i],tol=toler1)


# 		y_test      =   predict_clf(sgd_class,X_test[i])
# 		pa_score    =   accuracy_score(y_test,y_complete[i])
# 		pa_list.append(pa_score)

# 		y_train     =   predict_clf(sgd_class,folds_X_complete[i])
# 		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
# 		ta_list.append(ta_score)

# 		total_pa    =   total_pa + pa_score
# 		total_ta    =   total_ta + ta_score   

# 	pa = total_pa/5
# 	ta = total_ta/5

# 	return pa, ta, pa_list, ta_list


# tol1 = 0.01
# tol2 = 0.04

# for i in range(3):
	

# 	pa_sgd_score1, ta_sgd_score1, pa_sgd1, ta_sgd1 = sgd_tolerance(tol1)
# 	print("tolerance=")
# 	print(tol1)
# 	print("pa=")
# 	print(pa_sgd_score1)
# 	print("ta=")
# 	print(ta_sgd_score1)

# 	print("#######################SDG with varying tolerance ############# ")


# 	pa_sgd_score1, ta_sgd_score1, pa_sgd1, ta_sgd1 = sgd_tolerance(tol2)
# 	print("tolerance=")
# 	print(tol2)
# 	print("pa=")
# 	print(pa_sgd_score1)
# 	print("ta=")
# 	print(ta_sgd_score1)

# 	tol1 = tol1/10
# 	tol2 = tol2/10

# 	print("#######################SDG with varying tolerance ############# ")


# ##################################################################### Train and Check Decision Tree Classifier ###############################################

def decision_tree_depth(depth):
	pa_list     = []
	ta_list     = []
	total_ta    = 0
	total_pa    = 0

	for i in range(5):
		dt_class   =   train_decision_tree(folds_X_complete[i],folds_y_complete[i],max_depth=depth)


		y_test      =   predict_clf(dt_class,X_test[i])
		pa_score    =   accuracy_score(y_test,y_complete[i])
		pa_list.append(pa_score)

		y_train     =   predict_clf(dt_class,folds_X_complete[i])
		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
		ta_list.append(ta_score)

		total_pa    =   total_pa + pa_score
		total_ta    =   total_ta + ta_score   

	pa = total_pa/5
	ta = total_ta/5

	return pa, ta, pa_list, ta_list

# print("#######################Decision tree with varying depth ############# ")

pa_dt_score, ta_dt_score, pa_dt, ta_dt = decision_tree_depth(None)
pa_dt_list = pa_dt
# print ("depth: unconstrained")
# print("pa = ")
# print(pa_dt_score)
# print("ta = ")
# print(ta_dt_score)


# print("#######################Decision tree with varying depth ############# ")

pa_dt_score1, ta_dt_score1, pa_dt1, ta_dt1 = decision_tree_depth(4)
pa_dt_list4 = pa_dt1
# print("depth: 4")
# print("pa = ")
# print(pa_dt_score1)
# print("ta = ")
# print(ta_dt_score1)

# print("#######################Decision tree with varying depth ############# ")


pa_dt_score1, ta_dt_score1, pa_dt1, ta_dt1 = decision_tree_depth(8)
pa_dt_list8 = pa_dt1
# print("depth:8")
# print("pa = ")
# print(pa_dt_score1)
# print("ta = ")
# print(ta_dt_score1)



# ########################################################### Train and SGD  Stumps ############################################################3


def sgd_Stumps():

	pa_list     = []
	ta_list     = []
	total_ta    = 0
	total_pa    = 0
	a           = 0
	b           = 0



	for i in range(5):

		a,b   =   train_sgd_with_stumps(folds_X_complete[i],folds_y_complete[i])


		y_test      =   predict_sgd_with_stumps(a,b,X_test[i])
		pa_score    =   accuracy_score(y_test,y_complete[i])
		pa_list.append(pa_score)

		y_train     =   predict_sgd_with_stumps(a,b,folds_X_complete[i])
		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
		ta_list.append(ta_score)

		total_pa    =   total_pa + pa_score
		total_ta    =   total_ta + ta_score   

	pa = total_pa/5
	ta = total_ta/5

	return pa, ta, pa_list, ta_list




pa_stumps_score, ta_stumps_score, pa_stumps, ta_stumps = sgd_Stumps()
pa_stumps_list = pa_stumps

# print("#######################SDG with stumps_feature ############# ")
# print("pa = ")
# print(pa_stumps_score)
# print(pa_stumps)
# print("ta = ")
# print(ta_stumps_score)
# print(ta_stumps)



# ############################################ Calculating Confidence ##################################################################
# import math 
# import numpy as np 


# def calc_confidence(pa):

# 	x_bar   = np.mean(pa)
# 	S       = np.std(pa)
# 	n       = len(pa)

# 	ci_lower = x_bar - 4.604*(S/math.sqrt(n))
# 	ci_upper = x_bar + 4.604*(S/math.sqrt(n))

# 	return [ci_upper, ci_lower]


# ci_for_dt       =   calc_confidence(pa_dt_list)
# ci_for_dt4      =   calc_confidence(pa_dt_list4)
# ci_for_dt8      =   calc_confidence(pa_dt_list8)
# ci_for_sgd      =   calc_confidence(pa_sgd_list)
# ci_for_stumps   =   calc_confidence(pa_stumps_list)
# print("confidence interval for decision tree with unconstrained depth= ")
# print(ci_for_dt)
# print("\n")
# print("confidence interval for decision tree with depth 4= ")
# print(ci_for_dt4)
# print("\n")
# print("confidence interval for decision tree with depth 8= ")
# print(ci_for_dt8)
# print("\n")
# print("confidence interval for Stochastic Gradient Descent= ")
# print(ci_for_sgd)
# print("\n")
# print("confidence interval for Stochastic Gradient Descent with stumps= ")
# print(ci_for_stumps)


# ############################################ Extracting unlabeled Data and full train data ##############################################################

unlabel_data_X         = []
unlabel_name_feature   = []
unlabel_name_list      = []

 
with open('test.unlabeled.txt' , 'r') as f:
	for line in f:
		content     = line.strip()
		word        = content.split()
		unlabel_name_list.append(word[0]+" " + word[1]+ " " + word[2])
				 
	for name in unlabel_name_list:
		unlabel_name_feature   =   compute_features(name)
		unlabel_data_X.append(unlabel_name_feature)



X_train_full = []
y_train_full = []

name_feature_full   =   []
name_list_full      =   []

with open('train.txt' , 'r') as f:
		for line in f:
			content = line.strip()
			word = content.split()
			name_list_full.append(word[1]+ " " + word[2] + " " + word[3])
			if word[0] == '+':
				y_train_full.append(1)
			else:
				y_train_full.append(0)  
		for name in name_list_full:
			name_feature_full =compute_features(name)
			X_train_full.append(name_feature_full)
#print(X_train_full,y_train_full)


# ###################################### Testing unlabelled data and writing results #########################################################3



# ######################################SGD and Decision Tree ########################################################


def unlabeled_prediction(classifier):

	y_unlabeld          = predict_clf(classifier,unlabel_data_X)
	y_unlabeld_list     = []
	for label in y_unlabeld:
		if(label==1):
			y_unlabeld_list.append('+')
		else:
			y_unlabeld_list.append('-')

	return y_unlabeld_list


y_unlabel_sgd   =   unlabeled_prediction(train_sgd(X_train_full,y_train_full,alpha = 0.001))
y_unlabel_dt    =   unlabeled_prediction(train_decision_tree(X_train_full,y_train_full))
y_unlabel_dt4   =   unlabeled_prediction(train_decision_tree(X_train_full,y_train_full,4))
y_unlabel_dt8   =   unlabeled_prediction(train_decision_tree(X_train_full,y_train_full,8))



with open("sgd.txt", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in y_unlabel_sgd:
		writer.writerow([val])
with open("dt.txt", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in y_unlabel_dt:
		writer.writerow([val])        
with open("dt-4.txt", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in y_unlabel_dt4:
		writer.writerow([val])
with open("dt-8.txt", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in y_unlabel_dt8:
		writer.writerow([val])
	 
# #print(y_unlabel_sgd)       


# ############################## SGD Stumps ######################################

def unlabeled_stumps():
	te, xe =train_sgd_with_stumps(X_train_full,y_train_full)
	y_unlabel_stumps = predict_sgd_with_stumps(te, xe, unlabel_data_X)
	y_final_stumps = []
	for label in y_unlabel_stumps:
		if(label == 1):
			y_final_stumps.append('+')
		elif(label == 0):
			y_final_stumps.append('-')
	return y_final_stumps



y_fin_unlabel_sgdt = unlabeled_stumps()
with open("sgd-dt.txt", "w") as output:
	writer = csv.writer(output, lineterminator='\n')
	for val in y_fin_unlabel_sgdt:
		writer.writerow([val])



