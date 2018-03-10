import tensorflow as tf 
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# what exactly goes on here:
# raw input --> weight it --> through hidden layer 1(activation 1) --> weight it --> through hidden layer 2(activation 2) --> weight it --> output
# (feed forward network)
# compare the output to intended output based on cost function 
# optimization (optimizer) minimizes the loss (Adam optimizer,gd,etc)
#backpropogation and manipulate values of weights
#one cycle=epoch =feed forward + backpropogation

from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("/tmp/data/", one_hot =True)

#3 layers in the neural network and number of nodes in each layer
n_nodes_hl1 =500
n_nodes_hl2	=500
n_nodes_hl3	=500
n_classes=10 #number of different outputs
batch_size=100 #number of data samples loaded into network at once

x=tf.placeholder('float',[None,784])
y=tf.placeholder('float')

#below function is the model of our network
def neural_network_model(data):
	hidden_1_layer={'weight':tf.Variable(tf.random_normal([784,n_nodes_hl1])),'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
	hidden_2_layer={'weight':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
	hidden_3_layer={'weight':tf.Variable(tf.random_normal([n_nodes_hl2,n_nodes_hl3])),'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}
	output_layer  ={'weight':tf.Variable(tf.random_normal([n_nodes_hl3,n_classes])),'biases':tf.Variable(tf.random_normal([n_classes]))}	

	l1=tf.add(tf.matmul(data,hidden_1_layer['weight']),hidden_1_layer['biases'])
	l1=tf.nn.relu(l1) #relu is max(0,x)
	
	l2=tf.add(tf.matmul(l1,hidden_2_layer['weight']),hidden_2_layer['biases'])
	l2=tf.nn.relu(l2)
	
	l3=tf.add(tf.matmul(l2,hidden_3_layer['weight']),hidden_3_layer['biases'])
	l3=tf.nn.relu(l3)
	
	output=tf.add(tf.matmul(l3,output_layer['weight']),output_layer['biases'])

	return output

def train_neural_network(x):
	prediction=neural_network_model(x)
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y)) #the cost function
	optim=tf.train.AdamOptimizer().minimize(cost)
	hm_epochs=10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss=0
			for l in range(int(mnist.train.num_examples/batch_size)):
				X, Y=mnist.train.next_batch(batch_size)
				c, l=sess.run([cost,optim],feed_dict={x: X,y: Y})
				epoch_loss+=c
			print('Epoch:',epoch, 'completed:',hm_epochs,'loss:',epoch_loss)
	
		# predict=prediction.eval({x:X})
		# print(predict)	
		correct=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
		accuracy=tf.reduce_mean(tf.cast(correct,'float'))
		print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

train_neural_network(x)


#in one epoch 100 images are loaded into the network 