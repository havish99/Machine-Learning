import tensorflow as tf 
import numpy as np 
init_op = tf.global_variables_initializer()

b=tf.placeholder(tf.float32,[None,1])
a=[1,2,3,4,5]
a=np.array(a)
a=a.reshape(-1,1)
with tf.Session() as sess:
	sess.run(init_op)
	output=sess.run(b,feed_dict={b: a})

print(output)