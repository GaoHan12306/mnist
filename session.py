# Author:Han
# @Time : 2019/3/15 15:40

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 创建一张图包含一组op和tensor,上下文环境
# op:只要使用tensorflow的API定义的函数都是OP
# tensor :就指代是数据

# 实现一个加法运算
a = tf.constant(5.0)
b = tf.constant(6.0)
sum1 = tf.add(a,b)
print(sum1)
# 默认的这张图，相当于是给程序分配一段内存
graph = tf.get_default_graph()
print(graph)

# 不是op不能使用会话的run()运行
var1 = 2.0
# var2 = 3.0
# sum2 = var1 + var2

# 有重载的机制，Tonsor默认会给运算符重载成op类型
sum2 = var1 + a
print(sum2)


# 只能运行一个图 默认（graph），但可以在会话当中指定图去运行
# 只要有会话的上下文环境 就可以方便的用eval()，可以使用tf.interactiveSession()

# 训练模型
# 实时的提供数据区进行训练

# placehoder()是一个占位符op，feed_dict是一个字典
plt = tf.placeholder(tf.float32, [None, 3])

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # print(sum1.eval())
    print(sess.run(plt,feed_dict={plt:[[1,2,3],[4,5,6],[1,2,3],[4,5,6]]
                                  }))
    print(sess.run([a,b,sum1]))
    print(sess.run(sum2)) # 进行了重载
    print(a.graph)
    print(sess.graph)
