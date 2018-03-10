import tensorflow as tf 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' #idk what this means but it sure avoids warnings
a=tf.constant(5)
b=tf.constant(6)

result= tf.multiply(a,b)

print(result)

sess=tf.Session()

print(sess.run(result))

sess.close()