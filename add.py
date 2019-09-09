# Author:Han
# @Time : 2019/3/15 14:34

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)



with tf.Session() as sess:
    print(sess.run(sum1))
