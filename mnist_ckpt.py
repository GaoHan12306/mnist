# Author:Han
# @Time : 2019/3/17 17:50

# Author:Han
# @Time : 2019/3/17 9:31
"""
tf.nn.conv2d(input, filter, strides=, padding=, name=None)
计算给定4-Dinput和  filter张量的2维卷积
input：给定的输入张量，具有[batch,heigth,width,
channel]，类型为float32,64
filter：指定过滤器的大小，[filter_height, filter_width,
       in_channels, out_channels]
strides：strides = [1, stride, stride, 1],步长
"""
"""
tf.nn.max_pool(value, ksize=, strides=, padding=,name=None)
输入上执行最大池数
value:4-D Tensor形状[batch, height, width, channels]
ksize:池化窗口大小，[1, ksize, ksize, 1]
strides:步长大小，[1,strides,strides,1]
padding:“SAME”, “VALID”，使用的填充算法的类型，
使用“SAME”

"""

import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from tensorflow.examples.tutorials.mnist import input_data


FLAGS = tf.app.flags.FLAGS  # tf.app.flags.FLAGS可以从对应的命令行参数取出参数

tf.app.flags.DEFINE_integer("is_train", 1, "指定程序是预测还是训练")


# 定义一个初始化权重的函数
def weight_vaeriables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w

# 定义一个初始化偏置的函数
def bias_vaeriables(shape):
    b = tf.Variable(tf.constant(0.0, shape=shape))
    return b



def model():
    """
    自定义的卷积模型
    :return:
    """
    # 1、准备数据的占位符 x [None,784] y_true[None,18]
    with tf.variable_scope("data"):
        x = tf.placeholder(tf.float32, [None, 784])
        y_true = tf.placeholder(tf.float32, [None, 10])

    # 2、一卷积层 卷积5*5*1,32个，步长=1   激活:tf.nn.relu  池化
    with tf.variable_scope("conv1"):
        # 随机初始化过滤层权重,以及偏置[32]
        w_conv1 = weight_vaeriables([5, 5, 1, 32])
        b_conv1 = bias_vaeriables([32])

        # 对x进行形状的改变[None,784]==>[None,28,28,1]
        x_reshape = tf.reshape(x, [-1, 28, 28, 1])

        # [None,28,28,1]----->[None,28,28,32]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshape, w_conv1, strides=[1, 1, 1, 1], padding="SAME") + b_conv1)

        # 池化 2*2，strides2 [None,28,28,32]----->[None,14,14,32]
        x_pool1 = tf.nn.max_pool(x_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 3、二卷积层 卷积5*5*32 64个filter strides=1 激活：tf.nn.relu  池化：
    with tf.variable_scope("conv2"):
        # 随机过滤层权重，权重[5,5,32,64] 偏置[64]
        w_conv2 = weight_vaeriables([5, 5, 32, 64])
        b_conv2 = bias_vaeriables([64])
        # 卷积[None,14,14,32]------->[None,14,14,64] 激活
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, strides=[1, 1, 1, 1], padding="SAME") + b_conv2)

        # 池化2*2 stride2 [None,14,14,64]----->[None,7,7,64]

        x_pool2 = tf.nn.max_pool(x_relu2,ksize=[1,2,2,1] , strides=[1, 2, 2, 1], padding="SAME")


    # 4、全连接层 [None,7,7,64]---->[None,7*7*64]*[7*7*64,10]=[None,10]
    with tf.variable_scope("fc"):
        # 随机初始化权重和偏置
        w_fc = weight_vaeriables([7 * 7 * 64, 10])
        b_fc = bias_vaeriables([10])

        # 高纬度变量收集
        tf.summary.histogram("weightes", w_fc)
        tf.summary.histogram("biases", b_fc)

        # 修改形状[None,7,7,64]---->[None,7*7*64]

        x_fc_reshape = tf.reshape(x_pool2, [-1, 7 * 7 * 64])


        # 进行矩阵运算得出 每个样本的10个结果 [None,10]
        y_predict = tf.matmul(x_fc_reshape, w_fc) + b_fc

    return x, y_true, y_predict


def conv_fc():
    # 获取真实的数据
    mnist = input_data.read_data_sets("./data/mnist/input_data/", one_hot=True)
    # 定义模型，得出输出(所以有现成的模型直接调用 即可)
    x, y_true, y_predict = model()

    # 进行交叉熵损失计算
    # 3、求出所有样本的损失，然后求平均值
    with tf.variable_scope("soft_cross"):

        # 求平均交叉熵损失
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_predict))

    # 4、梯度下降求出损失
    with tf.variable_scope("optimizer"):

        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)  # 卷积神经网络的学习率很低

    # 5、计算准确率
    with tf.variable_scope("acc"):

        equal_list = tf.equal(tf.argmax(y_true, 1), tf.argmax(y_predict, 1))
        # equal_list  None个样本   [1, 0, 1, 0, 1, 1,..........]
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    # tensorboard 的可视化获取过程
    # 收集变量 单个数值的收集
    tf.summary.scalar("losses",loss)
    tf.summary.scalar("acc", accuracy)
    # 收集变量 高纬度变量收集
    # tf.summary.histogram("weightes", weight)
    # tf.summary.histogram("biases", bias)

    # 定义一个合并变量的op
    merge = tf.summary.merge_all()

    # 定义一个初始化op
    init_op = tf.global_variables_initializer()

    # 创建一个saver
    saver = tf.train.Saver()

    # 开启会话运行
    with tf.Session() as sess:
        sess.run(init_op)

        # 建立events文件，然后写入
        filewriter = tf.summary.FileWriter("./mnistTensorboard/",graph=sess.graph)

        if FLAGS.is_train == 1:

            # 循环训练
            for i in range(1000):
                # 取出真实存在的特征值和目标值
                mnist_x, mnist_y = mnist.train.next_batch(50)
                # 运行train_op训练
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})

                # 写入每步训练的值到tensorboard中
                summary = sess.run(merge,feed_dict={x: mnist_x, y_true: mnist_y})
                filewriter.add_summary(summary,i)

                print("训练第%d步,准确率为:%f" % (i, sess.run(accuracy, feed_dict={x: mnist_x, y_true: mnist_y})))

            # 训练完后保存模型
            saver.save(sess, "./ckpt/fc_model")
        else:
            # 加载模型
            saver.restore(sess, "./ckpt/fc_model")

            # 如果是0，做出预测
            for i in range(100):
                # 每次测试一张图片 [0,0,0,0,0,1,0,0,0,0]
                x_test, y_test = mnist.test.next_batch(1)

                print("第%d张图片，手写数字图片目标是:%d, 预测结果是:%d" % (
                    i,
                    tf.argmax(y_test, 1).eval(), # 找出最大数字的行号 然后取出tensor的值
                    tf.argmax(sess.run(y_predict, feed_dict={x: x_test, y_true: y_test}), 1).eval()
                ))

    return None


if __name__ == "__main__":
    conv_fc()
