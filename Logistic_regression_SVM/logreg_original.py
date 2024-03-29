'''
	TEMPLATE FOR MACHINE LEARNING HOMEWORK
	AUTHOR Eric Eaton
'''

import numpy as np

class LogisticRegression:

	def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.00000001, maxNumIters = 2000):

		'''
		Constructo
		Arguments:
			alpha is the learning rate
			regLambda is the regularization parameter
			regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
			epsilon is the convergence parameter
			maxNumIters is the maximum number of iterations to run
		'''


		self.alpha = alpha
		self.regNorm = regNorm
		self.regLambda = regLambda
		self.epsilon = epsilon
		self.max_Iter = maxNumIters
		self.theta = None
		self.mean = None
		self.std = None

		

	

	def computeCost(self, theta, X, y, regLambda):

		'''
		Computes the objective function
		Arguments:
			X is a n-by-d numpy matrix
			y is an n-dimensional numpy vector
			regLambda is the scalar regularization constant
		Returns:
			a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
		'''


		yhat = self.sigmoid(np.matmul(X,theta))

		Sub = 1-yhat

		Sub[np.where(Sub==0)]=1e-10
		yhat[np.where(yhat==0)]=1e-10

		if self.regNorm == 1:
			cost = -(np.matmul(y.T,np.log(yhat)) + np.matmul((1-y).T,np.log(Sub))) + (regLambda/2)*np.linalg.norm(theta,1)

		elif self.regNorm == 2:
			cost = -(np.matmul(y.T,np.log(yhat)) + np.matmul((1-y).T,np.log(Sub))) + (regLambda/2)*np.linalg.norm(theta,2)

		return cost







		

	
	
	def computeGradient(self, theta, X, y, regLambda):
		'''
		Computes the gradient of the objective function
		Arguments:
			X is a n-by-d numpy matrix
			y is an n-dimensional numpy vector
			regLambda is the scalar regularization constant
		Returns:
			the gradient, an d-dimensional vector
		'''
		yhat 	= 	self.sigmoid(np.matmul(X,theta))

		lambd_mat 		= 	regLambda * np.identity(len(theta))
		lambd_mat[0,0] 	= 	0


		regul = np.matmul(lambd_mat,theta)
		mat   = np.matmul(X.T,(yhat-y))

		grad = mat + regul

		return grad
		

		

		
	


	def fit(self, X, y):
		'''
		Trains the model
		Arguments:
			X is a n-by-d numpy matrix
			y is an n-dimensional numpy vector

		'''
		iterations = 1

		
		

		n,d = X.shape
		X = np.c_[np.ones((n,1)),X]


		theta_old = np.random.normal(0, 0.1, (d+1))
		theta_old = theta_old[np.newaxis]
		theta_old = theta_old.T
		 


		theta_curr  = np.random.normal(0, 0.1, (d+1))
		theta_curr  = theta_curr[np.newaxis]
		theta_curr = theta_curr.T

		while(self.hasConverged(theta_curr,theta_old)):

			theta_old = theta_curr

			grad 	= 	self.computeGradient(theta_curr,X,y,self.regLambda)
			

			theta_curr = theta_curr - self.alpha * grad 


			cost = self.computeCost(theta_curr,X,y,self.regLambda)

			if np.isnan(cost[0,0]):
				print("nan cost ")
				print(X)
				print(y)
				break

			iterations 		= 	iterations + 1

			print("Iteration: ", iterations,"cost:", cost, " Theta: ", theta_curr)

			if (iterations>=self.max_Iter):
				print("maximum iterations reached")
				break
			# break
		if (~self.hasConverged(theta_curr,theta_old)):
			print("we have convergence")
		
		self.theta = theta_curr




			



	def predict(self, X):
		'''
		Used the model to predict values for each instance in X
		Arguments:
			X is a n-by-d numpy matrix
		Returns:
			an n-dimensional numpy vector of the predictions
		-'''

		
		n,d = X.shape
		X = np.c_[np.ones((n,1)),X]

		y_pred = self.sigmoid(np.matmul(X,self.theta))

		for i in range(y_pred.shape[0]):
			if y_pred[i]>0.5:
				y_pred[i]=1
			else:
				y_pred[i]=0

		return np.array(y_pred)



## make binary the ypred you get 

	def sigmoid(self, Z):

		val =  1/ (  1+np.exp(-Z) )
		val = np.array(val)
		# val = np.rint(val)
		#print (val)
		return val
		'''
		Computes the sigmoid function 1/(1+exp(-z))
		'''
	def hasConverged(self, theta_new,theta_old):
		

		if np.linalg.norm(theta_new-theta_old) > self.epsilon:

			return True
		else:
			return False








# def data():
# 	Y_set   =   []
# 	X_set   =   []
# 	with open('data1.dat' , 'r') as f:
# 		for line in f:
# 			line_content = line.strip()
# 			content_word = line_content.split(',')
# 			content_word = [float(i) for i in content_word]
# 			content_word = np.array(content_word)
# 			X_set.append ([content_word[0],content_word[1]])
# 			Y_set.append(content_word[2])
# 		return(X_set,Y_set)





# Log_ob = LogisticRegression()

# X,Y = data()
# X = np.array(X)
# Y = np.array(Y)
# X = np.c_[np.ones((100,1)),X]


