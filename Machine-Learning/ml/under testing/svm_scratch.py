#cvxopt for convex optimisation
import pandas as pd 
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np
style.use('ggplot')

class Support_Vector_Machine:
	def __init__(self,visualisation=True):				#self is common variable in class
		self.visualisation=visualisation
		self.colors={1:'r' -1:'b'}
		if self.visualisation:
			self.fig-plt.figure()
			self.ax=self.fig.add_subplot(1,1,1)

	#training
	def fit(self,data):
		self.data=data
		#dict of magnitude of w as key and w,b as the values 
		opt_dict={}
		all_data=[]
		transforms=[[1,1],[-1,1],[-1,-1],[1,-1]]
		for yi in self.data:
			for featureset in self.data[yi]:
				for features in featureset:
					all_data.append(features)
		self.max_data=max(all_data)
		self.min_data=min(all_data)
		
		all_data=None



	#predicting	
	def predict(self,features):
		classification=np.sign(np.dot(np.array(features),self.w)+self.b)
		return classification








data_dict={-1:np.array([[1,7],[2,8],[3,8]]),1:np.array([[5,1],[6,-1],[7,3]])}

