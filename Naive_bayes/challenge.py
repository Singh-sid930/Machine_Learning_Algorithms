import csv 
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt 

import plotly.plotly as py
import plotly.graph_objs as go


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

from sklearn.metrics import precision_recall_curve



def data_train():
	Y_set   =   []
	X_set   =   []
	with open('challengeTrainLabeled.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
			else:
				X_set.append(row[0:len(row)-1])
				Y_set.append(row[len(row)-1])
				line_count += 1
		return(X_set,Y_set)




def clean_data(X):

	X_gender = X[:,0]
	X_race2=X[:,1]
	X_race2[np.where(X_race2=='nan')]='7'
	X_age = X[:,2]
	X_BP = X[:,3]
	X_BP[np.where(X_BP=='nan')]='-1'
	X_MIL = X[:,4]
	X_MIL[np.where(X_MIL=='nan')]='-1'
	X_pulse_stat = X[:,5]
	X_pulse_stat[np.where(X_pulse_stat=='nan')]='-1'
	X_abd_dia = X[:,6]
	X_abd_dia[np.where(X_abd_dia=='nan')]='-1'
	X_preg = X[:,7]
	X_preg[np.where(X_preg=='nan')]='4'
	X_fl_wa= X[:,8]
	X_fl_wa[np.where(X_fl_wa=='nan')]='-1'
	X_fl_pa = X[:,9]
	X_fl_pa[np.where(X_fl_pa=='nan')]='-1'

	X_urine = X[:,10]
	X_urine[np.where(X_urine=='nan')]='-1'
	X_vi_B = X[:,11]
	X_vi_B[np.where(X_vi_B=='nan')]='3'
	X_vi_C = X[:,12]
	X_vi_C[np.where(X_vi_C=='nan')]='3'
	X_vi_D = X[:,13]
	X_vi_D[np.where(X_vi_D=='nan')]='3'
	X_haemo = X[:,14]
	X_haemo[np.where(X_haemo=='nan')]='-1'	
	X_hepa_A = X[:,15]
	X_hepa_A[np.where(X_hepa_A=='nan')]='3'
	X_hepa_D = X[:,16]
	X_hepa_D[np.where(X_hepa_D=='nan')]='3'
	X_hepa_B = X[:,17]
	X_hepa_B[np.where(X_hepa_B=='nan')]='3'
	X_calcium = X[:,18]
	X_calcium[np.where(X_calcium=='nan')]='-1'
	X_glucose = X[:,19]
	X_glucose[np.where(X_glucose=='nan')]='-1'
	X_iron = X[:,20]
	X_iron[np.where(X_iron=='nan')]='3'
	X_vi_E = X[:,21]
	X_vi_E[np.where(X_vi_E=='nan')]='3'


	X_Q4 = X[:,22]
	X_Q4[np.where(X_Q4=='nan')]='7'
	X_Q12 = X[:,23]
	X_Q12[np.where(X_Q12=='nan')]='7'
	X_Q14 = X[:,24]
	X_Q14[np.where(X_Q14=='nan')]='7'
	X_Q17 = X[:,25]
	X_Q17[np.where(X_Q17=='nan')]='7'
	X_Q18 = X[:,26]
	X_Q18[np.where(X_Q18=='nan')]='7'
	X_Q19 = X[:,27]
	X_Q19[np.where(X_Q19=='nan')]='7'
	X_Q20 = X[:,28]
	X_Q20[np.where(X_Q20=='nan')]='7'
	X_Q21 = X[:,29]
	X_Q21[np.where(X_Q21=='nan')]='7'
	X_Q22 = X[:,30]
	X_Q22[np.where(X_Q22=='nan')]='7'
	X_Q23= X[:,31]
	X_Q23[np.where(X_Q23=='nan')]='7'
	X_Q24= X[:,32]
	X_Q24[np.where(X_Q24=='nan')]='7'
	X_Q25= X[:,33]
	X_Q25[np.where(X_Q25=='nan')]='7'
	X_Q26= X[:,34]
	X_Q26[np.where(X_Q26=='nan')]='7'





	X_weight = X[:,35]
	X_weight[np.where(X_weight=='nan')]='0'
	X_weight[np.where(X_weight=='')]='0'
	X_weight = X_weight.astype(np.float)
	weight_mean = np.sum(X_weight)/len(X_weight)
	X_weight[np.where(X_weight==0)]=weight_mean



	


	X = np.array([X_gender,X_race2,X_age,X_BP,X_MIL,X_pulse_stat,X_abd_dia,X_preg,X_fl_wa,X_fl_pa,X_urine,X_vi_B,X_vi_E,X_vi_D,X_vi_C,X_haemo,X_hepa_B,X_hepa_D,X_hepa_A,X_calcium,X_glucose,X_iron,X_Q4,X_Q12,X_Q14,X_Q17,X_Q18,X_Q19,X_Q20,X_Q21,X_Q22,X_Q23,X_Q24,X_Q25,X_Q26,X_weight])

	return X 



def svm_clf(X_test,X,y):
    # using the classifier 
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv=5)

    clf.fit(X,y)

    y_predict = clf.predict(X_test)
    
    return (y_predict,np.mean(scores))

def dec_clf(X_test,X,y):
    
    # usign the classfier 
    
    clf = DecisionTreeClassifier(criterion ='gini',random_state=200)
    scores = cross_val_score(clf, X, y, cv=5)

    clf.fit(X,y)

    y_predict = clf.predict(X_test)
    
    return (y_predict,np.mean(scores))

def random_forest(X_test,X,y):

    clf=RandomForestClassifier(n_estimators=100, max_depth=5,random_state=200)
    
    scores = cross_val_score(clf, X,y,cv=5)
    
    clf.fit(X,y)

    y_predict = clf.predict(X_test)
    
    return (y_predict,np.mean(scores))
    
def ada_clf(X_test,X,y):
    
        
    #clf =svm.SVC(gamma='auto',probability= True)
    
    clf = DecisionTreeClassifier(max_depth = 1,criterion ='gini',random_state=200)
    
    clf_ada = AdaBoostClassifier(n_estimators=300,base_estimator=clf,learning_rate=1)
   
    scores = cross_val_score(clf_ada, X,y,cv=5)
    
    clf_ada.fit(X,y)

    y_predict = clf_ada.predict(X_test)
    
    return (y_predict,np.mean(scores))

def bagging(X_test,X,y):	
    
        
    #clf =svm.SVC(gamma='auto',probability= True)
    
    clf = DecisionTreeClassifier(max_depth = 1,criterion ='gini',random_state=200)
    
    clf_bag = BaggingClassifier(n_estimators=500,base_estimator=clf)

    scores = cross_val_score(clf_bag, X,y,cv=5)

    clf_bag.fit(X,y)

    y_predict = clf_bag.predict(X_test)
    
    return (y_predict,np.mean(scores))



#********************************************* VISUALIZING DATA **************************************************************###


# import seaborn as sns
# import missingno as msno

# df = pd.read_csv('challengeTrainLabeled.csv')
# df.info()
# sns.heatmap(df.isnull(), cbar=False)


#*********************************************Extracting test and train data *************************************************####
 



def data_test():
	Y_set   =   []
	X_set   =   []
	with open('challengeTestUnlabeled.csv') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		line_count = 0
		for row in csv_reader:
			if line_count == 0:
				line_count += 1
				
			else:
				X_set.append(row[0:len(row)-1])
				Y_set.append(row[len(row)-1])
				line_count += 1
		return(X_set,Y_set)


selec_words = np.array([4,7,23,56,57,58,59,62,63,64,65,66,67,68,69,70,73,74,75,76,77,78,79,32,41,43,46,47,48,49,50,51,52,53,54,55,61])


## ******************** TRaining data ***************************************#

X,Y = data_train()
X_train = np.array(X)
Y_train = np.array(Y)


for i in range(len(X_train)):
	X_train[i] = pd.to_numeric(X_train[i],errors='coerce')


X_train_pick = X_train[:,selec_words]

X_train_clean = clean_data(X_train_pick)

X_train = np.transpose(X_train_clean)
[r,c] = X_train.shape


X_train = X_train.astype(np.float)

for i in range(len(X_train)):
	X_train[i,:] = X_train[i,:].astype(np.float)


# print(X_train.shape)
# print(Y_train.shape) 

##********************** Test data ************************************#

X,Y = data_test()
X_test = np.array(X)
Y_test = np.array(Y)

for i in range(len(X_test)):
	X_test[i] = pd.to_numeric(X_test[i],errors='coerce')


X_test_pick = X_test[:,selec_words]

X_test_clean = clean_data(X_test_pick)

X_test = np.transpose(X_test_clean)
[r,c] = X_test.shape


X_test = X_test.astype(np.float)

for i in range(len(X_test)):
	X_test[i,:] = X_test[i,:].astype(np.float)


# print(X_test.shape)
# print(Y_test.shape)


#******************************************** Training Prediction and output ******************************************************##

y_pred,score = bagging(X_test,X_train,Y_train)
print(score)


y_pred,score = ada_clf(X_test,X_train,Y_train)
print(score)


y_pred,score = random_forest(X_test,X_train,Y_train)
print(score)


y_pred,score = dec_clf(X_test,X_train,Y_train)
print(score)

X = np.array(X)
id_pat = X[:,[1]]

id_pat = id_pat.astype(np.float)

id_pat = id_pat.reshape((len(id_pat),))

a = np.array([id_pat,y_pred])

a = a.astype(np.float)

a = np.transpose(a)


# # print(a)

# You will have to include the heading titles at the top manually. 

np.savetxt("res.csv", a, delimiter=",")
