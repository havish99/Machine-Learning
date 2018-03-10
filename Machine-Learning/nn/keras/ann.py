import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense
white=pd.read_csv('winequality-white.csv',sep=';')

red=pd.read_csv('winequality-red.csv',sep=';')

white['type']=0
red['type']=1
wines=red.append(white,ignore_index=True)

#corr=wines.corr() correlation matrix

#sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)

#plt.show()

X=np.array(wines.ix[:,0:11])
X=preprocessing.scale(X)
y=np.array(wines['type'])
k=0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33)
print(X_train)
print(len(X_train))
model = Sequential()

# input layer 
model.add(Dense(12, activation='relu', input_shape=(11,)))

# one hidden layer 
model.add(Dense(8, activation='relu'))
# output layer 
model.add(Dense(1, activation='sigmoid'))
# Model output shape
print(model.output_shape)

# Model summary
print(model.summary())

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=5, batch_size=4, verbose=1)

y_predict=model.predict(X_test)

print(y_predict)
