from __future__ import print_function
import numpy as np
from builtins import range
import numpy as np

#按照卷积核进行数据映射 ####
def split_by_strides(x, filter_size):
    N, C, H, W = x.shape
    oh = (H - filter_size) // 1 + 1
    ow = (W - filter_size) // 1 + 1
    strides = (*x.strides[:-2], x.strides[-2] * 1, x.strides[-1] * 1, *x.strides[-2:])
    A = np.lib.stride_tricks.as_strided(x, shape=(N, C, oh, ow, filter_size, filter_size), strides=strides)
    return A

#前向 卷积 ####
def conv_forward(x, w, b, conv_param):
    # x_col, w_col, H_out, W_out = im2col(x, w, conv_param)
    # out = Conv(x_col, w_col, b, H_out, W_out)

    s, p = conv_param['stride'], conv_param['pad']
    filter_size = conv_param['filter_size']

    if p > 0:
        x = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')  # 补零

    conv_out = split_by_strides(x, filter_size)
    conv_out = np.tensordot(conv_out, w, axes=([1, 4, 5], [1, 2, 3])) + b
    conv_out = conv_out.reshape(conv_out.shape[0], conv_out.shape[3], conv_out.shape[1], conv_out.shape[2])
    cache = (x, w, b, conv_param)
    return conv_out, cache

# 前向 relu ####
def relu_forward(x):
    out = np.maximum(0, x)
    cache = x
    return out, cache

#前向 池化####
def max_pool_forward_reshape(x, pool_param):
    N, C, H, W = x.shape
    pool_height, pool_width = pool_param['pool_height'], pool_param['pool_width']
    stride = pool_param['stride']
    assert pool_height == pool_width == stride, 'Invalid pool params'
    assert H % pool_height == 0
    assert W % pool_height == 0
    x_reshaped = x.reshape(N, C, H // pool_height, pool_height,
                           W // pool_width, pool_width)
    out = x_reshaped.max(axis=3).max(axis=4)

    cache = (x, x_reshaped, out)
    return out, cache

# 反向 池化####
def max_pool_backward_reshape(dout, cache):
    x, x_reshaped, out = cache

    dx_reshaped = np.zeros_like(x_reshaped)
    out_newaxis = out[:, :, :, np.newaxis, :, np.newaxis]
    mask = (x_reshaped == out_newaxis)
    dout_newaxis = dout[:, :, :, np.newaxis, :, np.newaxis]
    dout_broadcast, _ = np.broadcast_arrays(dout_newaxis, dx_reshaped)
    dx_reshaped[mask] = dout_broadcast[mask]
    dx_reshaped /= np.sum(mask, axis=(3, 5), keepdims=True)
    dx = dx_reshaped.reshape(x.shape)

    return dx

#卷积——relu——池化######
def Conv_relu_pool_forward(x, w, b, conv_param, pool_param):  #gamma, beta, esp

    conv_out, conv_cache = conv_forward(x, w, b, conv_param)
    relu_out, relu_cache = relu_forward(conv_out)
    pool_out, pool_cache = max_pool_forward_reshape(relu_out, pool_param)
    # pool_out, pool_cache = pool_forward(relu_out, pool_param)
    cache = (conv_cache, relu_cache, pool_cache)
    return pool_out, cache

#全连接####
def affine_forward(x, w, b):
    out = None
    x_N = x.reshape(x.shape[0], -1)
    out = x_N.dot(w) + b
    cache = (x, w, b)
    return out, cache

####
def softmax_loss(x, y):
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

#反向 全连接层####
def affine_backward(dout, cache):

    x, w, b = cache
    dx, dw, db = None, None, None

    db = np.sum(dout, axis=0)
    x_N = x.reshape(x.shape[0], -1)
    dw = np.dot(x_N.T, dout)
    dx = np.dot(dout, w.T)
    dx = dx.reshape(x.shape)

    return dx, dw, db

#反向 relu
def relu_backward(dout, cache):
    dx, x = None, cache
    dx = dout
    dx[x < 0] = 0
    return dx

#反向 卷积 ####
def conv_backward_naive2(dout, cache):

    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    filter_size = conv_param['filter_size']

    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_out = int(1 + (H + 2 * pad - HH) / stride)
    W_out = int(1 + (W + 2 * pad - WW) / stride)

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    # db = np.sum(dout, axis=(0, 2, 3))
    db = np.sum(dout.reshape(dout.shape[1], -1), 1)

    s = stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_pad = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dout_w = np.pad(dout, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_pad = dx_pad.astype('float64')

    xx = split_by_strides(x_pad, filter_size)
    dw = np.tensordot(xx, dout_w, axes=([0, 2, 3], [0, 2, 3]))
    dw = dw.reshape(dw.shape[3], dw.shape[0], dw.shape[1], dw.shape[2])

    dout_x = np.pad(dout, ((0, 0), (0, 0), (filter_size-1, filter_size-1), (filter_size-1, filter_size-1)), 'constant')
    dout_x = split_by_strides(dout_x, filter_size)
    dx = np.tensordot(dout_x, w, axes=([1, 4, 5], [0, 2, 3]))
    dx = dx.reshape(dx.shape[0], dx.shape[3], dx.shape[1], dx.shape[2])

    return dx, dw, db

#反向 池化，relu，卷积 ####
def conv_relu_pool_backward(dout, cache):

    conv_cache, relu_cache, pool_cache = cache
    ds = max_pool_backward_reshape(dout, pool_cache)
    da = relu_backward(ds, relu_cache)
    dx, dw, db = conv_backward_naive2(da, conv_cache)
    return dx, dw, db

# #反向传播--池化，relu，BN, 卷积
# def conv_relu_pool_batch_backward(dout, cache):

#     conv_cache, relu_cache, pool_cache, cache_conv_batchnorm = cache
#     ds = max_pool_backward_reshape(dout, pool_cache)
#     da = relu_backward(ds, relu_cache)

#     da, dgamma, dbeta = batchnorm_backward(da, cache_conv_batchnorm)
#     dx, dw, db = conv_backward_naive2(da, conv_cache)
#     return dx, dw, db

# #反向传播--池化，relu，卷积   ####
# def conv_relu_backward(dout, cache):
#     conv_cache, relu_cache = cache
#     da = relu_backward(dout, relu_cache)
#     # dx, dw, db = conv_backward_fast(da, conv_cache)
#     dx, dw, db = conv_backward_naive2(da, conv_cache)
#     return dx, dw, db
