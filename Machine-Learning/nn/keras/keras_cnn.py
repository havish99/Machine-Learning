#advantage in keras-no need to worry about output size
from __future__ import division

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential

(x_train,y_train),(x_test,y_test)=mnist.load_data()

batch_size=128
epochs=10
num_classes=10

img_x=28
img_y=28
# reshape the data to get a 4d tensor where each element constitutes the sample number, the image dimensions and the rgb components
# here, the images are all grey scale so the last component is 1
x_train=x_train.reshape(x_train.shape[0],img_x,img_y,1)
x_test=x_test.reshape(x_test.shape[0],img_x,img_y,1)

x_train=x_train.astype('float32')
x_test=x_test.astype('float32')

input_shape=(img_x,img_y,1)

#normalization step useful to prevent any outliers
x_train=x_train/255
x_test=x_test/255
#doing the one-hot conversion for the output
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)
# the model
model=Sequential()
# a layer which involves 32 filters each of size 5,5
model.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=input_shape))
# doing max pooling: chooses the maximum value in a 2x2 part of a big matrix
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# one more layer of convolution
model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu'))
# again max pooling
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# convert to 1d array to send it to MLP
model.add(Flatten())
#MLP
model.add(Dense(1000,activation='relu'))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1)

