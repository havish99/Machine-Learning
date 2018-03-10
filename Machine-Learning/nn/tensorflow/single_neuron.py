import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


init=tf.global_variables_initializer()
x=tf.constant(1.0)
y_=tf.constant(0.0)
w=tf.Variable(0.8)
y=x*w
loss=(y-y_)**2

optim=tf.train.AdamOptimizer().minimize(loss)

grad_val=optim.compute_gradients(loss)
sess=tf.Session()
sess.run(init)
l=sess.run(grad_val)
sess.close()
print(l)


