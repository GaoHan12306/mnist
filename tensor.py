# Author:Han
# @Time : 2019/3/15 20:01
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# tensor的形状 0 1 2 3阶数
# tensorflow:打印出来的形状表示
# print(tensor.shape)
# 控制台表示 0维：() 1维：(5) 2维：(5,6) 3维：(2,3,4)
# 形状：代码用[]表示

# 形状的概念
# 静态形状 动态形状
# 静态形状来说，一旦张量确定 不能修改
# 动态形状 可以创建一个新的张量来设定新的形状，可以跨维度。但元素的数量要相同
plt = tf.placeholder(tf.float32,[None,2]) # type shape name(可视化)
print(plt)

plt.set_shape([3, 2]) # 不能跨维度设定
print(plt)

# plt.set_shape([4, 2]) # 不能再修改
plt_reshape = tf.reshape(plt,[2,3]) # 动态形状改变的时候 元素数量一定要匹配
with tf.Session() as  sess:
    pass


