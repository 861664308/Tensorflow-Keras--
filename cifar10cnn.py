# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:22:17 2019

@author: 鲁金川
"""

#TensorFlow卷积神经网络识别cifar10数据集

#导入cifar10数据集, 标准化特征并且将label转换为one-hot编码
import tensorflow as tf
from keras.datasets import cifar10
import numpy as np
(x_img_train, y_label_train), (x_img_test, y_label_test) = cifar10.load_data()
L = x_img_train.shape[0]
x_img_train_normalize = x_img_train / 255.0
x_img_test_normalize = x_img_test / 255.0
y_label_train = tf.one_hot(y_label_train, 10, 1, 0)
y_label_test = tf.one_hot(y_label_test, 10, 1, 0)
with tf.Session() as sess:
  y_label_train = sess.run(y_label_train)
  y_label_test = sess.run(y_label_test)
#x_img_train_normalize = x_img_train_normalize.reshape(-1, 10000)
#x_img_test_normalize = x_img_test_normalize.reshape(-1, 10000)
#建立共享函数
#权重函数
def weight(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev = 0.1), 'W')
#偏置函数
def bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape), 'b')
#卷积运算
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')
#池化运算
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#建立next_batch函数
def next_batch(dataSet, batchsize, i):
    batch = dataSet[i * batchsize : min([batchsize * (i + 1), L - 1])]
    return batch
#建立模型
#建立输入层
with tf.name_scope('Input_Layer'):
    x = tf.placeholder('float', shape=[None, 32, 32, 3], name='x')
    x_image = x
#建立卷积层1
with tf.name_scope('C1_Conv'):
    W1 = weight([3, 3, 3, 32])
    b1 = bias([32])
    Conv1 = conv2d(x_image, W1) + b1
    C1_conv = tf.nn.relu(Conv1)
    C1_conv_dropout = tf.nn.dropout(C1_conv, keep_prob=0.75)
#建立池化层1
with tf.name_scope('C1_pool'):
    C1_pool = max_pool_2x2(C1_conv_dropout)
#建立卷积层2
with tf.name_scope('C2_Conv'):
    W2 = weight([3, 3, 32, 64])
    b2 = bias([64])
    Conv2 = conv2d(C1_pool, W2) + b2
    C2_conv = tf.nn.relu(Conv2)
    C2_conv_dropout = tf.nn.dropout(C2_conv, keep_prob=0.75)
#建立池化层2
with tf.name_scope('C2_pool'):
    C2_pool = max_pool_2x2(C2_conv_dropout)
#建立平坦层
with tf.name_scope('D_Flat'):
    D_Flat = tf.reshape(C2_pool, [-1, 8 * 8 * 64])
#建立隐藏层
with tf.name_scope('D_Hidden_Layer'):
    W3 = weight([8 * 8 * 64, 128])
    b3 = bias([128])
    D_Hidden = tf.nn.relu(tf.matmul(D_Flat, W3) + b3)
    D_Hidden_Dropout = tf.nn.dropout(D_Hidden, keep_prob=0.75)
#建立输出层
with tf.name_scope('Output_Layer'):
    W4 = weight([128, 10])
    b4 = bias([10])
    y_predict = tf.nn.softmax(tf.matmul(D_Hidden_Dropout, W4) + b4)
    
#定义训练方式
with tf.name_scope('optimizer'):
    y_label = tf.placeholder('float', shape=[None, 10], name='ylabel')
    loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=y_predict, labels=y_label))
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss_function)

#定义评估模型准确率的方式
with tf.name_scope('evaluate_model'):
    correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

#进行训练
#定义训练参数
trainEpochs = 15
batchSize = 128
if np.mod(L, batchSize) == 0:
    totalBatchs = int(x_img_train.shape[0] / batchSize)
else:
    totalBatchs = int(x_img_train.shape[0] / batchSize) + 1
epoch_list = [] ; accuracy_list = [] ; loss_list = []
from time import time
startTime = time()
#开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(trainEpochs):
        epochtime = time()
        for i in range(totalBatchs):
            batch_x = next_batch(x_img_train_normalize, batchSize, i)
            batch_y = next_batch(y_label_train, batchSize, i)
            #batch_x = sess.run(batch_x)
            batch_y = np.squeeze(batch_y)
            #batch_y = sess.run(batch_y)
            y_label_test = np.squeeze(y_label_test)
            #y_label_test = sess.run(y_label_test)
            sess.run(optimizer, feed_dict = {x : batch_x, y_label : batch_y})
            if (epoch % 100 == 0):
                print('epoch' + str(epoch + 1)  + ' batch' + str(i) + ' has finished!')
        loss, acc = sess.run([loss_function, accuracy], feed_dict = {x : x_img_test_normalize,
                             y_label : y_label_test})
        epoch_list.append(epoch)
        loss_list.append(loss)
        accuracy_list.append(acc)
        print('Train Epoch' + str(epoch + 1) + '：,Loss=' + str(loss) + 
              ', Accuracy=' + str(acc) + ', time=' + str(time() - epochtime) + 's')
    duration = time() - startTime
    print('Train Finished Takes:' + str(duration))