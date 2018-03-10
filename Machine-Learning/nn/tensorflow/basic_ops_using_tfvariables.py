import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
graph=tf.get_default_graph()
x=tf.constant(1.0)
w=tf.Variable(0.8)
out=x*w
op=graph.get_operations()

# for op in op:
# 	print(op.name)
sess=tf.Session()
init=tf.global_variables_initializer() #all operations are executed only inside a session. if there are variables, then this command must be used to initialize the variables
sess.run(init)
k=sess.run(out)
print(k)
sess.close()


