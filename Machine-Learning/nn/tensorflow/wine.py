import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn import cross_validation
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
df = pd.read_csv('wine.csv')
#code to convert non-numeric data to numeric data
x_train=df.drop(['class'],1)
one_hot=pd.get_dummies(df['class'])
df=df.drop('class',axis=1)
df=df.join(one_hot)
y_train=df.drop(x_train,1)
x_train=np.array(x_train)
y_train=np.array(y_train)
correct=0
#x_train, x_test, y_train, y_test=cross_validation.train_test_split(l,m,test_size=0.2)
X=tf.placeholder('float',[None,13])
Y=tf.placeholder('float',[None,3])
n_l1=64
n_l2=64
epochs=100
weights={'h1': tf.Variable(tf.random_normal([13,n_l1])),
		'h2': tf.Variable(tf.random_normal([n_l1,n_l2])),
		'out': tf.Variable(tf.random_normal([n_l2,3])),}

biases={'b1': tf.Variable(tf.random_normal([n_l1])),
		'b2': tf.Variable(tf.random_normal([n_l2])),
		'out': tf.Variable(tf.random_normal([3]))
}


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

x=np.array(list(chunks(x_train,89)))
y=np.array(list(chunks(y_train,89)))

def neural_network(x,weights,biases):
	l_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
	l_1=tf.nn.relu(l_1)
	l_2=tf.add(tf.matmul(l_1,weights['h2']),biases['b2'])
	l_2=tf.nn.relu(l_2)
	out=tf.add(tf.matmul(l_2,weights['out']),biases['out'])
	return out

predict=neural_network(X,weights,biases)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=Y))
optim = tf.train.AdamOptimizer(learning_rate=0.025).minimize(cost)
init=tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		loss=0
		for i in range(2):

			_, c=sess.run([optim,cost],feed_dict={X:x[i],Y:y[i]})
			loss=loss+c 
		print(loss,epoch)
	prediction=predict.eval({X:x_train})

#print(np.corrcoef(y_train[:, 0], prediction[:, 0])[0][1]*100)	
for i in range(len(prediction)):
	l=np.amax(prediction[i])
	for j in range(3):
		if(prediction[i][j]<l):
			prediction[i][j]=0
		if(prediction[i][j]==l):
			prediction[i][j]=1
print(prediction)

lol=prediction-y_train
print(lol)
for i in range(len(lol)):
	if(lol[i][0]==0 and lol[i][1]==0 and lol[i][2]==0):
		correct+=1

print(correct)


