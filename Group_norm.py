# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 18:18:56 2018

@author: 74284
"""
import tensorflow as tf
def GroupNorm(x,G=1,eps=1e-5):
    #x=tf.expand_dims(x,3)
    N,H,W,C=x.shape
    #print(x.shape)
    #x=tf.reshape(x,[N,G,C//G,H,W])
    mean,var=tf.nn.moments(x,[1,2,3],keep_dims=True)
    x=(x-mean)/tf.sqrt(var+eps)
    gamma = tf.Variable(tf.constant(1.0, shape=[C]), dtype=tf.float32, name='gamma')
    beta = tf.Variable(tf.constant(0.0, shape=[C]), dtype=tf.float32, name='beta')
    gamma = tf.reshape(gamma, [1, 1, 1,C])
    beta = tf.reshape(beta, [1, 1, 1,C])
    gamma=tf.cast(gamma, tf.float32)
    beta=tf.cast(beta, tf.float32)
    x=tf.cast(x,tf.float32)
    x_output = x * gamma + beta
    print(x_output.shape)
    #x_output=tf.reshape(x_output,[N,H,W])
    return x_output
