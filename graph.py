# Author:Han
# @Time : 2019/3/15 15:10
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一张图包含一组op和tensor,上下文环境
# op:只要使用tensorflow的API定义的函数都是OP
# tensor :就指代是数据
g = tf.Graph()
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)

# 实现一个加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a,b)

# 默认的这张图，相当于是给程序分配一段内存
graph = tf.get_default_graph()
print(graph)

with tf.Session() as sess:
    print(sess.run(sum1))
    print(a.graph)
    print(sess.graph)