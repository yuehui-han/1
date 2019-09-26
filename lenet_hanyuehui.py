from __future__ import print_function
from lenet_Function import *
from builtins import range
from builtins import object
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys
from tensorflow.examples.tutorials.mnist import input_data


input_dim=(1, 28, 28)  #输入图像尺寸
filter_size=5         #卷积核大小
num_classes=10        #类别数
reg = 0.0001

params = {}
batchnorm_params1 = {}
batchnorm_params2 = {}
C, H, W = input_dim

#损失函数
def loss1(X, y=None):
    W1, b1 = params['W1'], params['b1']
    W2, b2 = params['W2'], params['b2']
    W3, b3 = params['W3'], params['b3']
    W4, b4 = params['W4'], params['b4']
    W5, b5 = params['W5'], params['b5']

    filter_size = W1.shape[2]
    conv_param1 = {'stride': 1, 'pad': 2, 'filter_size': filter_size}
    conv_param2 = {'stride': 1, 'pad': 0, 'filter_size': filter_size}
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    if y is None:     # 测试时
        #第一层卷积层，relu，池化层
        conv_forward_out_1, cache_forward_1 = Conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)

        #第二层卷积层，relu，池化层
        conv_forward_out_2, cache_forward_2 = Conv_relu_pool_forward(conv_forward_out_1, W2, b2, conv_param2, pool_param)
        
        #全连接层 relu
        affine_forward_out_3, cache_forward_3 = affine_forward(conv_forward_out_2, W3, b3)
        affine_relu_3, cache_relu_3 = relu_forward(affine_forward_out_3)

        #全连接层 relu
        affine_forward_out_4, cache_forward_4 = affine_forward(affine_relu_3, W4, b4)
        affine_relu_4, cache_relu_4 = relu_forward(affine_forward_out_4)
        
        #输出层
        scores, cache_forward_5 = affine_forward(affine_relu_4, W5, b5)
        return scores

    else:    # 训练时
        #第一层卷积层，relu，池化层
        conv_forward_out_1, cache_forward_1 = Conv_relu_pool_forward(X, W1, b1, conv_param1, pool_param)

        # 第二层卷积层，relu，池化层
        conv_forward_out_2, cache_forward_2 = Conv_relu_pool_forward(conv_forward_out_1, W2, b2, conv_param2, pool_param)

        # 全连接层 relu
        affine_forward_out_3, cache_forward_3 = affine_forward(conv_forward_out_2, W3, b3)
        affine_relu_3, cache_relu_3 = relu_forward(affine_forward_out_3)

        # 全连接层 relu
        affine_forward_out_4, cache_forward_4 = affine_forward(affine_relu_3, W4, b4)
        affine_relu_4, cache_relu_4 = relu_forward(affine_forward_out_4)

        # 输出层
        scores, cache_forward_5 = affine_forward(affine_relu_4, W5, b5)

        loss, grads = 0, {}
        loss, dout = softmax_loss(scores, y)

        loss += reg * 0.5 * (np.sum(pow(params['W1'], 2)) + np.sum(pow(params['W2'], 2)) + np.sum(pow(params['W3'], 2)) + np.sum(
            pow(params['W4'], 2)) + np.sum(pow(params['W5'], 2)))

        
        # 反向传播 
        #全连接层
        dX5, grads['W5'], grads['b5'] = affine_backward(dout, cache_forward_5)
  
        # relu 全连接层
        dX4 = relu_backward(dX5, cache_relu_4)
        dX4, grads['W4'], grads['b4'] = affine_backward(dX4, cache_forward_4)

        # relu 全连接层
        dX3 = relu_backward(dX4, cache_relu_3)
        dX3, grads['W3'], grads['b3'] = affine_backward(dX3, cache_forward_3)

        # 卷积 relu 池化
        dX2, grads['W2'], grads['b2'] = conv_relu_pool_backward(dX3, cache_forward_2)

        # 卷积 relu 池化
        dX1, grads['W1'], grads['b1'] = conv_relu_pool_backward(dX2, cache_forward_1)

        #参数更新
        grads['W5'] = grads['W5'] + reg * grads['W5']
        grads['W4'] = grads['W4'] + reg * grads['W4']
        grads['W3'] = grads['W3'] + reg * grads['W3']
        grads['W2'] = grads['W2'] + reg * grads['W2']
        grads['W1'] = grads['W1'] + reg * grads['W1']
        return loss, grads


#预测函数
def predict(X):
    scores = loss1(X)
    y_pred = np.argmax(scores, axis=1)
    return y_pred


#训练函数
def training(X, y, X_val, y_val, X_test, y_test, learning_rate=1e-3, learning_rate_decay=0.95,
           num_iters=2000, batch_size=20):

    num_train = X.shape[0]
    iterations_per_epoch = int(max(num_train // batch_size, 1))

    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_iters):
        # num_batch = 0
        for per_epoch in range(iterations_per_epoch):
            X_batch = None
            y_batch = None

            X_batch = X[per_epoch * batch_size:(per_epoch + 1) * batch_size, :, :, :]
            y_batch = y[per_epoch * batch_size:(per_epoch + 1) * batch_size]

            loss, grads = loss1(X_batch, y_batch)
            loss_history.append(loss)

            if per_epoch % 10 == 0:
                train_acc = sum(np.array(predict(X_batch)) == np.array(y_batch.astype('int64'))) / X_batch.shape[0]
                val_acc = sum(np.array(predict(X_val)) == np.array(y_val.astype('int64'))) / y_val.shape[0]

                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)


                print('iteration %d / %d: loss %f --- train_acc %f  val_acc %f ' % (epoch, per_epoch, loss, train_acc, val_acc))

            params['W5'] += -learning_rate * grads['W5']
            params['b5'] += -learning_rate * grads['b5']

            params['W4'] += -learning_rate * grads['W4']
            params['b4'] += -learning_rate * grads['b4']

            params['W3'] += -learning_rate * grads['W3']
            params['b3'] += -learning_rate * grads['b3']

            params['W2'] += -learning_rate * grads['W2']
            params['b2'] += -learning_rate * grads['b2']

            params['W1'] += -learning_rate * grads['W1']
            params['b1'] += -learning_rate * grads['b1']

        Test_acc = sum(np.array(predict(X_test)) == np.array(y_test.astype('int64'))) / y_test.shape[0]
        print('Test acc is %f' % (Test_acc))

        learning_rate *= learning_rate_decay


if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data2/", one_hot=True)
    X_train, Y_train = mnist.train.images, mnist.train.labels #训练集
    X_val,  Y_val = mnist.validation.images, mnist.validation.labels #验证集
    X_test, Y_test = mnist.test.images, mnist.test.labels   #测试集

    #格式，尺寸转换
    X_train = X_train.astype('float64')
    X_val = X_val.astype('float64')
    X_test = X_test.astype('float64')
    X_train = X_train.reshape(55000, 1, 28, 28)
    X_val = X_val.reshape(5000, 1, 28, 28)
    X_test = X_test.reshape(10000, 1, 28, 28)

    Y_tr = np.argmax(Y_train, axis=1)
    Y_val = np.argmax(Y_val, axis=1)
    Y_te = np.argmax(Y_test, axis=1)

    # 参数初始化
    weight_scale = 0.1
    params['W1'] = weight_scale * np.random.randn(32, 1, 5, 5)  
    params['b1'] = np.zeros(32)

    params['W2'] = weight_scale * np.random.randn(64, 32, 5, 5) 
    params['b2'] = np.zeros(64)

    params['W3'] = weight_scale * np.random.randn(1600, 512) 
    params['b3'] = np.zeros(512)

    params['W4'] = weight_scale * np.random.randn(512, 128)
    params['b4'] = np.zeros(128)

    params['W5'] = weight_scale * np.random.randn(128, 10)
    params['b5'] = np.zeros(10)


    training(X_train, Y_tr, X_val, Y_val, X_test, Y_te,
             learning_rate=0.01, learning_rate_decay=0.99,
             num_iters=50, batch_size=100)

