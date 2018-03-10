import numpy as np
import pandas as pd
import tensorflow as tf
import os
from sklearn import cross_validation
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
df = pd.read_csv('abalone.csv')
categor = ['sex']
#code to convert non-numeric data to numeric data
for c in categor:
    types = np.unique(df[c])
    id_maker = {v: k for k, v in enumerate(types)}
    df[c] = [id_maker[x] for x in df[c]]
x_train=np.array(df.drop(['class'],1))
y_train=np.array(df['class']).reshape(-1,1)
#x_train, x_test, y_train, y_test=cross_validation.train_test_split(l,m,test_size=0.2)
X=tf.placeholder('float',[None,8])
Y=tf.placeholder('float',[None,1])
n_l1=64
n_l2=64
epochs=1000
print(len(x_train))
weights={'h1': tf.Variable(tf.random_normal([8,n_l1])),
		'h2': tf.Variable(tf.random_normal([n_l1,n_l2])),
		'out': tf.Variable(tf.random_normal([n_l2,1])),}

biases={'b1': tf.Variable(tf.random_normal([n_l1])),
		'b2': tf.Variable(tf.random_normal([n_l2])),
		'out': tf.Variable(tf.random_normal([1]))
}


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

x=np.array(list(chunks(x_train,90)))
y=np.array(list(chunks(y_train,90)))
def neural_network(x,weights,biases):
	l_1=tf.add(tf.matmul(x,weights['h1']),biases['b1'])
	l_1=tf.nn.relu(l_1)
	l_2=tf.add(tf.matmul(l_1,weights['h2']),biases['b2'])
	l_2=tf.nn.relu(l_2)
	out=tf.add(tf.matmul(l_2,weights['out']),biases['out'])
	return out

predict=neural_network(X,weights,biases)
cost=tf.reduce_mean(tf.squared_difference(Y,predict))
optim = tf.train.AdamOptimizer(learning_rate=0.025).minimize(cost)
init=tf.global_variables_initializer()





with tf.Session() as sess:
	sess.run(init)
	for epoch in range(epochs):
		loss=0
		for i in range(38):

			_, c=sess.run([optim,cost],feed_dict={X:x[i],Y:y[i]})
			loss=loss+c 
		print(loss,epoch)
	prediction=predict.eval({X:x_train})

#print(np.corrcoef(y_train[:, 0], prediction[:, 0])[0][1]*100)	
print(prediction)






