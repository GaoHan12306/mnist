# Author:Han
# @Time : 2019/3/15 20:40

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建全部为0 的张量
zero = tf.zeros([2,3],dtype=tf.float32,name=None)
print(zero)
print(tf.shape(zero))
# 创建随机正态分布数据 mean表示分布平均值,stddev标准差=表示分布幅度
height = tf.truncated_normal([10],mean=160,stddev=10,dtype=tf.float32,seed=None,name=None)

# 张量数据类型的转化
a = tf.constant(5)
print(a)
print(tf.cast(a,tf.float32))

# 张量的扩展
b = [[1, 2, 3], [4, 5, 6]]
c = [[7,8,9],[10,11,12]]
sum = tf.concat([b,c],axis=1) #0按照行扩展 1按照列扩展

with tf.Session() as sess:
    # print(sess.run(zero))
    print(zero.eval())
    print(height.eval())
    print(sum.eval())





