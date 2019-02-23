import numpy as np
from logreg_adagrad import LogisticRegressionAdagrad

def data_set():
	Y_set   =   []
	X_set   =   []
	name_list    =   []

	with open('retinopathy.dat', 'r') as f:
		for line in f:
			line_content = line.strip()
			content_word = line_content.split(',')

			n = len(content_word)
			if content_word[n-1] == '0':
				Y_set.append(0)
			elif content_word[n-1] =='1':
				Y_set.append(1)
	print(len(Y_set))

	with open('retinopathy.dat', 'r') as f:		   
		for line in f:
			line_content = line.strip()
			content_word = line_content.split(',')
			n = len(content_word)
			X_set.append(content_word[0:n-2])
	print(len(X_set))


	X_set = np.array(X_set)
	X_set = X_set.astype(float)
	Y_set = np.array(Y_set)
	Y_set = Y_set.astype(float)
	Y_set = Y_set[np.newaxis]
	Y_set = Y_set.T

	return(X_set,Y_set)



X,y = data_set()
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

total = np.append(X,y,1)

X0,X1,X2,X3,X4 = np.array_split(total,5)

y1 = X1[:,len(X1[0])-1][np.newaxis].T
X1= X1[:,0:len(X1[0])-1]

y2 = X2[:,len(X2[0])-1][np.newaxis].T
X2 = X2[:,0:len(X2[0])-1]

y3 = X3[:,len(X3[0])-1][np.newaxis].T
X3 = X3[:,0:len(X3[0])-1]

y4 = X4[:,len(X4[0])-1][np.newaxis].T
X4 = X4[:,0:len(X4[0])-1]

y0 = X0[:,len(X0[0])-1][np.newaxis].T
X0 = X0[:,0:len(X0[0])-1]



X_complete = [X0,X1,X2,X3,X4]
y_complete = [y0,y1,y2,y3,y4]


foldX_0 = np.concatenate((X1,X2,X3,X4),axis=0)
foldy_0 = np.concatenate((y1,y2,y3,y4),axis=0)
test0   = X0

foldX_1 = np.concatenate((X0,X2,X3,X4),axis=0)
foldy_1 = np.concatenate((y0,y2,y3,y4),axis=0)
test1   = X1

foldX_2 = np.concatenate((X0,X1,X3,X4),axis=0)
foldy_2 = np.concatenate((y0,y1,y3,y4),axis=0)
test2   = X2

foldX_3 = np.concatenate((X0,X1,X2,X4),axis=0)
foldy_3 = np.concatenate((y0,y1,y2,y4),axis=0)
test3   = X3

foldX_4= np.concatenate((X0,X1,X2,X3),axis=0)
foldy_4 = np.concatenate((y0,y1,y2,y3),axis=0)
test4   = X4



# # In[ ]:

folds_X_complete    = [foldX_0,foldX_1,foldX_2,foldX_3,foldX_4]
folds_y_complete    = [foldy_0,foldy_1,foldy_2,foldy_3,foldy_4]
X_test              = [X0,X1,X2,X3,X4]





from sklearn.metrics import accuracy_score
import pandas 
import csv


# #####################################################Train and check SDG Classifier #################################################################

# ######################### SDG With varying alpha #######################

def logregA_varying_regularization(alpha,regul1):
	pa_list     = []
	ta_list     = []
	total_ta    = 0
	total_pa    = 0

	for i in range(5):

		Log_ob = LogisticRegressionAdagrad(alpha = alpha,regNorm = regul1)

		Log_ob.fit(folds_X_complete[i],folds_y_complete[i])

		y_test 		=   Log_ob.predict(X_test[i])
		pa_score    =   accuracy_score(y_test,y_complete[i])
		pa_list.append(pa_score)

		y_train     =   Log_ob.predict(folds_X_complete[i])
		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
		ta_list.append(ta_score)

		total_pa    =   total_pa + pa_score
		total_ta    =   total_ta + ta_score   

	pa = total_pa/5
	ta = total_ta/5

	return pa, ta, pa_list, ta_list


# print("printing SGD classifier values ")

pa_sgd_score1, ta_sgd_score1, pa_sgd1, ta_sgd1 = logregA_varying_regularization(0.1,1)
pa_sgd_score2, ta_sgd_score2, pa_sgd2, ta_sgd2 = logregA_varying_regularization(0.001,1)
pa_sgd_score3, ta_sgd_score3, pa_sgd3, ta_sgd3 = logregA_varying_regularization(1,1)
pa_sgd_score4, ta_sgd_score4, pa_sgd4, ta_sgd4 = logregA_varying_regularization(0.12,2)
pa_sgd_score5, ta_sgd_score5, pa_sgd5, ta_sgd5 = logregA_varying_regularization(0.002,2)
pa_sgd_score6, ta_sgd_score6, pa_sgd6, ta_sgd6 = logregA_varying_regularization(0.8,2)



print("Accuracy score for retino data set with norm 1 and alpha 0.1")
print("test accuracy list over the 5 folds =", pa_sgd1)
print("Training accuracy list over the 5 folds= ", ta_sgd1)
print("Average test accurace score = ",pa_sgd_score1)
print("Average training accurace score = ",ta_sgd_score1)

print("######################################################################3")


print("Accuracy score for retino data set with norm 1 and alpha 0.001")
print("test accuracy list over the 5 folds =", pa_sgd2)
print("Training accuracy list over the 5 folds= ", ta_sgd2)
print("Average test accurace score = ",pa_sgd_score2)
print("Average training accurace score = ",ta_sgd_score2)

print("######################################################################3")


print("Accuracy score for retino data set with norm 1 and alpha 1")
print("test accuracy list over the 5 folds =", pa_sgd3)
print("Training accuracy list over the 5 folds= ", ta_sgd3)
print("Average test accurace score = ",pa_sgd_score3)
print("Average training accurace score = ",ta_sgd_score3)

print("######################################################################3")


print("Accuracy score for retino data set with norm 2 and alpha 0.1")
print("test accuracy list over the 5 folds =", pa_sgd4)
print("Training accuracy list over the 5 folds= ", ta_sgd4)
print("Average test accurace score = ",pa_sgd_score4)
print("Average training accurace score = ",ta_sgd_score4)

print("######################################################################3")


print("Accuracy score for retino data set with norm 2 and alpha 0.001")
print("test accuracy list over the 5 folds =", pa_sgd5)
print("Training accuracy list over the 5 folds= ", ta_sgd5)
print("Average test accurace score = ",pa_sgd_score5)
print("Average training accurace score = ",ta_sgd_score5)

print("######################################################################3")


print("Accuracy score for retino data set with norm 2 and alpha 1")
print("test accuracy list over the 5 folds =", pa_sgd6)
print("Training accuracy list over the 5 folds= ", ta_sgd6)
print("Average test accurace score = ",pa_sgd_score6)
print("Average training accurace score = ",ta_sgd_score6)

print("######################################################################3")

# alp1 = 0.001
# alp2 = 0.005


# # for i in range(3):

# # 	pa_sgd_score, ta_sgd_score, pa_sgd, ta_sgd = sgd_alpha(alp1)
# # 	print("aplha = ")
# # 	print (alp1)
# # 	print("pa= ")
# # 	print(pa_sgd_score)
# # 	print("TA= ")
# # 	print(ta_sgd_score)
	

# # 	print("#######################SDG with varying Alpha print ############# ")

# # 	pa_sgd_score, ta_sgd_score, pa_sgd, ta_sgd = sgd_alpha(alp2)

# # 	print("aplha = ")
# # 	print (alp2)
# # 	print("pa= ")
# # 	print(pa_sgd_score)
# # 	print("TA= ")
# # 	print(ta_sgd_score)

# # 	print("#######################SDG with varying Alpha print ############# ")

	

# # 	alp1 = alp1/10
# # 	alp2 = alp2/10

# # ################################# SDG with varying tolerance #############


# # def sgd_tolerance(toler1):
# # 	pa_list     = []
# # 	ta_list     = []
# # 	total_ta    = 0
# # 	total_pa    = 0

# # 	for i in range(5):
# # 		sgd_class   =   train_sgd(folds_X_complete[i],folds_y_complete[i],tol=toler1)


# # 		y_test      =   predict_clf(sgd_class,X_test[i])
# # 		pa_score    =   accuracy_score(y_test,y_complete[i])
# # 		pa_list.append(pa_score)

# # 		y_train     =   predict_clf(sgd_class,folds_X_complete[i])
# # 		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
# # 		ta_list.append(ta_score)

# # 		total_pa    =   total_pa + pa_score
# # 		total_ta    =   total_ta + ta_score   

# # 	pa = total_pa/5
# # 	ta = total_ta/5

# # 	return pa, ta, pa_list, ta_list


# # tol1 = 0.01
# # tol2 = 0.04

# # for i in range(3):
	

# # 	pa_sgd_score1, ta_sgd_score1, pa_sgd1, ta_sgd1 = sgd_tolerance(tol1)
# # 	print("tolerance=")
# # 	print(tol1)
# # 	print("pa=")
# # 	print(pa_sgd_score1)
# # 	print("ta=")
# # 	print(ta_sgd_score1)

# # 	print("#######################SDG with varying tolerance ############# ")


# # 	pa_sgd_score1, ta_sgd_score1, pa_sgd1, ta_sgd1 = sgd_tolerance(tol2)
# # 	print("tolerance=")
# # 	print(tol2)
# # 	print("pa=")
# # 	print(pa_sgd_score1)
# # 	print("ta=")
# # 	print(ta_sgd_score1)

# # 	tol1 = tol1/10
# # 	tol2 = tol2/10

# # 	print("#######################SDG with varying tolerance ############# ")


# # ##################################################################### Train and Check Decision Tree Classifier ###############################################

# def decision_tree_depth(depth):
# 	pa_list     = []
# 	ta_list     = []
# 	total_ta    = 0
# 	total_pa    = 0

# 	for i in range(5):
# 		dt_class   =   train_decision_tree(folds_X_complete[i],folds_y_complete[i],max_depth=depth)


# 		y_test      =   predict_clf(dt_class,X_test[i])
# 		pa_score    =   accuracy_score(y_test,y_complete[i])
# 		pa_list.append(pa_score)

# 		y_train     =   predict_clf(dt_class,folds_X_complete[i])
# 		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
# 		ta_list.append(ta_score)

# 		total_pa    =   total_pa + pa_score
# 		total_ta    =   total_ta + ta_score   

# 	pa = total_pa/5
# 	ta = total_ta/5

# 	return pa, ta, pa_list, ta_list

# # print("#######################Decision tree with varying depth ############# ")

# pa_dt_score, ta_dt_score, pa_dt, ta_dt = decision_tree_depth(None)
# pa_dt_list = pa_dt
# # print ("depth: unconstrained")
# # print("pa = ")
# # print(pa_dt_score)
# # print("ta = ")
# # print(ta_dt_score)


# # print("#######################Decision tree with varying depth ############# ")

# pa_dt_score1, ta_dt_score1, pa_dt1, ta_dt1 = decision_tree_depth(4)
# pa_dt_list4 = pa_dt1
# # print("depth: 4")
# # print("pa = ")
# # print(pa_dt_score1)
# # print("ta = ")
# # print(ta_dt_score1)

# # print("#######################Decision tree with varying depth ############# ")


# pa_dt_score1, ta_dt_score1, pa_dt1, ta_dt1 = decision_tree_depth(8)
# pa_dt_list8 = pa_dt1
# # print("depth:8")
# # print("pa = ")
# # print(pa_dt_score1)
# # print("ta = ")
# # print(ta_dt_score1)



# # ########################################################### Train and SGD  Stumps ############################################################3


# def sgd_Stumps():

# 	pa_list     = []
# 	ta_list     = []
# 	total_ta    = 0
# 	total_pa    = 0
# 	a           = 0
# 	b           = 0



# 	for i in range(5):

# 		a,b   =   train_sgd_with_stumps(folds_X_complete[i],folds_y_complete[i])


# 		y_test      =   predict_sgd_with_stumps(a,b,X_test[i])
# 		pa_score    =   accuracy_score(y_test,y_complete[i])
# 		pa_list.append(pa_score)

# 		y_train     =   predict_sgd_with_stumps(a,b,folds_X_complete[i])
# 		ta_score    =   accuracy_score(y_train,folds_y_complete[i])
# 		ta_list.append(ta_score)

# 		total_pa    =   total_pa + pa_score
# 		total_ta    =   total_ta + ta_score   

# 	pa = total_pa/5
# 	ta = total_ta/5

# 	return pa, ta, pa_list, ta_list




# pa_stumps_score, ta_stumps_score, pa_stumps, ta_stumps = sgd_Stumps()
# pa_stumps_list = pa_stumps

# # print("#######################SDG with stumps_feature ############# ")
# # print("pa = ")
# # print(pa_stumps_score)
# # print(pa_stumps)
# # print("ta = ")
# # print(ta_stumps_score)
# # print(ta_stumps)



# # ############################################ Calculating Confidence ##################################################################
# # import math 
# # import numpy as np 


# # def calc_confidence(pa):

# # 	x_bar   = np.mean(pa)
# # 	S       = np.std(pa)
# # 	n       = len(pa)

# # 	ci_lower = x_bar - 4.604*(S/math.sqrt(n))
# # 	ci_upper = x_bar + 4.604*(S/math.sqrt(n))

# # 	return [ci_upper, ci_lower]


# # ci_for_dt       =   calc_confidence(pa_dt_list)
# # ci_for_dt4      =   calc_confidence(pa_dt_list4)
# # ci_for_dt8      =   calc_confidence(pa_dt_list8)
# # ci_for_sgd      =   calc_confidence(pa_sgd_list)
# # ci_for_stumps   =   calc_confidence(pa_stumps_list)
# # print("confidence interval for decision tree with unconstrained depth= ")
# # print(ci_for_dt)
# # print("\n")
# # print("confidence interval for decision tree with depth 4= ")
# # print(ci_for_dt4)
# # print("\n")
# # print("confidence interval for decision tree with depth 8= ")
# # print(ci_for_dt8)
# # print("\n")
# # print("confidence interval for Stochastic Gradient Descent= ")
# # print(ci_for_sgd)
# # print("\n")
# # print("confidence interval for Stochastic Gradient Descent with stumps= ")
# # print(ci_for_stumps)


# # ############################################ Extracting unlabeled Data and full train data ##############################################################

# unlabel_data_X         = []
# unlabel_name_feature   = []
# unlabel_name_list      = []

 
# with open('test.unlabeled.txt' , 'r') as f:
# 	for line in f:
# 		content     = line.strip()
# 		word        = content.split()
# 		unlabel_name_list.append(word[0]+" " + word[1]+ " " + word[2])
				 
# 	for name in unlabel_name_list:
# 		unlabel_name_feature   =   compute_features(name)
# 		unlabel_data_X.append(unlabel_name_feature)



# X_train_full = []
# y_train_full = []

# name_feature_full   =   []
# name_list_full      =   []

# with open('train.txt' , 'r') as f:
# 		for line in f:
# 			content = line.strip()
# 			word = content.split()
# 			name_list_full.append(word[1]+ " " + word[2] + " " + word[3])
# 			if word[0] == '+':
# 				y_train_full.append(1)
# 			else:
# 				y_train_full.append(0)  
# 		for name in name_list_full:
# 			name_feature_full =compute_features(name)
# 			X_train_full.append(name_feature_full)
# #print(X_train_full,y_train_full)


# # ###################################### Testing unlabelled data and writing results #########################################################3



# # ######################################SGD and Decision Tree ########################################################


# def unlabeled_prediction(classifier):

# 	y_unlabeld          = predict_clf(classifier,unlabel_data_X)
# 	y_unlabeld_list     = []
# 	for label in y_unlabeld:
# 		if(label==1):
# 			y_unlabeld_list.append('+')
# 		else:
# 			y_unlabeld_list.append('-')

# 	return y_unlabeld_list


# y_unlabel_sgd   =   unlabeled_prediction(train_sgd(X_train_full,y_train_full,alpha = 0.001))
# y_unlabel_dt    =   unlabeled_prediction(train_decision_tree(X_train_full,y_train_full))
# y_unlabel_dt4   =   unlabeled_prediction(train_decision_tree(X_train_full,y_train_full,4))
# y_unlabel_dt8   =   unlabeled_prediction(train_decision_tree(X_train_full,y_train_full,8))



# with open("sgd.txt", "w") as output:
# 	writer = csv.writer(output, lineterminator='\n')
# 	for val in y_unlabel_sgd:
# 		writer.writerow([val])
# with open("dt.txt", "w") as output:
# 	writer = csv.writer(output, lineterminator='\n')
# 	for val in y_unlabel_dt:
# 		writer.writerow([val])        
# with open("dt-4.txt", "w") as output:
# 	writer = csv.writer(output, lineterminator='\n')
# 	for val in y_unlabel_dt4:
# 		writer.writerow([val])
# with open("dt-8.txt", "w") as output:
# 	writer = csv.writer(output, lineterminator='\n')
# 	for val in y_unlabel_dt8:
# 		writer.writerow([val])
	 
# # #print(y_unlabel_sgd)       


# # ############################## SGD Stumps ######################################

# def unlabeled_stumps():
# 	te, xe =train_sgd_with_stumps(X_train_full,y_train_full)
# 	y_unlabel_stumps = predict_sgd_with_stumps(te, xe, unlabel_data_X)
# 	y_final_stumps = []
# 	for label in y_unlabel_stumps:
# 		if(label == 1):
# 			y_final_stumps.append('+')
# 		elif(label == 0):
# 			y_final_stumps.append('-')
# 	return y_final_stumps



# y_fin_unlabel_sgdt = unlabeled_stumps()
# with open("sgd-dt.txt", "w") as output:
# 	writer = csv.writer(output, lineterminator='\n')
# 	for val in y_fin_unlabel_sgdt:
# 		writer.writerow([val])



