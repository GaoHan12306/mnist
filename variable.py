# Author:Han
# @Time : 2019/3/15 21:45
# 变量 也是一种OP，是一种特殊的张量，能够进行 存储持久化 ，它的值就是张量。
# 变量必须初始化
# tf.global_variables_initializer() 添加一个初始化所有 变量 的op 在会话中开启

# 变量op
# 1、变量op能够持久化保存，普通张量op是不行的
# 2、当定义一个变量op的时候，一定要在会话中进行初始化
# 3、name的参数：在tensorboard使用的时候显示名字，可以区分相同的op
# tensorboard --logdir=E:\PycharmProject\02-23-深度学习day04\summary\test


import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

a = tf.constant(2,name="a")
b = tf.constant(3,name="b")
c = tf.add(a,b)
var = tf.Variable(tf.random_normal([2,3],mean=0.0,stddev=1.0),name="variable1")
print(a, var)
# 变量var 必须做一步显示的初始化 并在会话中开启
init_op = tf.global_variables_initializer()

with tf.Session() as  sess:
    # 必须初始化变量
    sess.run(init_op)

    # 把程序的图结构 序列化 写入事件文件， graph:把指定的图写进事件文件当中
    filewrite = tf.summary.FileWriter("./summary/test/",graph=sess.graph)

    print(sess.run([c ,var]))


