import pandas as pd 
import quandl
import math,datetime
import matplotlib.pyplot as plt 
from matplotlib import style
import numpy as np 
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

style.use('ggplot')
df = quandl.get('WIKI/GOOGL')
#cross_validation useful to split the data as we want
#print(df.head()) for printing the headers
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]#choosing required features

df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close'] * 100

df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open'] * 100 

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']] 

df.fillna(-99999, inplace=True)

forecast_col='Adj. Close'

forecast_out=int(math.ceil(0.01*len(df)))

df['label']=df[forecast_col].shift(-forecast_out)


x_total=np.array(df.drop(['label'],1)) #all columns except label column
x_total=preprocessing.scale(x_total)
x=x_total[:-forecast_out]
x_non=x_total[-forecast_out:]
df.dropna(inplace=True)
y=np.array(df['label'])
x_train, x_test, y_train, y_test=cross_validation.train_test_split(x,y,test_size=0.2)

clf=LinearRegression(n_jobs=-1)#classification algo used n_jobs is number of operations at the same time refer documentation
clf.fit(x_train,y_train)#trains the data
accuracy=clf.score(x_test,y_test)#tests the data
forecast_predict=clf.predict(x_non)#used to predict values

print(forecast_predict,accuracy,forecast_out)
#print(accuracy)

#basically this breaks down like this:
