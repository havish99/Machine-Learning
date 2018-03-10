import numpy as np 
from sklearn import preprocessing,cross_validation,linear_model,neighbors
import pandas as pd 


def cleanup(df):
	a=df.columns.values 
	df=df.drop(a[[0,3,8]],1)
	a=df.columns.values
	df.fillna(0,inplace=True)
#the following lines of code is helpful to convert non numeric data to numeric data
	for column in a:
 		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
 			content=set(df[column].values)
			content=list(content)
			a1=df[column].values
			for i in range(0,len(a1)):
				a1[i]=content.index(a1[i])
			df[column]=a1	
	return df
def cleanup1(df):
	a=df.columns.values 
	df=df.drop(a[[0,2,7]],1)
	a=df.columns.values
	df.fillna(0,inplace=True)
#the following lines of code is helpful to convert non numeric data to numeric data
	for column in a:
 		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
 			content=set(df[column].values)
			content=list(content)
			a1=df[column].values
			for i in range(0,len(a1)):
				a1[i]=content.index(a1[i])
			df[column]=a1	
	return df		
df=pd.read_csv('train.csv')

df=cleanup(df)

X=np.array(df.drop(['Survived'],1))

Y=np.array(df['Survived'])
#x_train, x_test, y_train, y_test=cross_validation.train_test_split(X,Y,test_size=0.1)
clf=linear_model.LogisticRegression()

clf.fit(X,Y)

df=pd.read_csv('test.csv')

df=cleanup1(df)

output=clf.predict(np.array(df))
print(len(output))
index=[]
a=892
for i in range(0,len(output)):

	print str(a)+","+str(output[i])
	a=a+1



 




