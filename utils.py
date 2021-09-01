"""
Author: Maosen Li, Shanghai Jiao Tong University
"""

import tensorflow as  tf
import numpy as np

def unif_weight_init(shape, name=None):

    initial = tf.random.uniform(shape, minval=-np.sqrt(6.0/(shape[0]+shape[1])), maxval=np.sqrt(6.0/(shape[0]+shape[1])), dtype=tf.float32)

    return tf.Variable(initial, name=name)


def sample_gaussian(mean, diag_cov):
    batch = tf.shape(diag_cov)[0]
    dim = tf.shape(diag_cov)[1]
    epsilon = tf.random.normal(shape=(batch, dim))
    # return z_mean + tf.exp(0.5 * z_log_var) * epsilon

    # z = mean+tf.random.normal(diag_cov.shape)*diag_cov
    z = mean+epsilon*diag_cov

    return z


def sample_gaussian_np(mean, diag_cov):

    z = mean+np.random.normal(size=diag_cov.shape)*diag_cov
    
    return z


def gcn_layer_id(norm_adj_mat, W, b):

    return tf.nn.relu(tf.add(tf.sparse.sparse_dense_matmul(norm_adj_mat, W), b))


def gcn_layer(norm_adj_mat, h, W, b):

    return tf.add(tf.matmul(tf.sparse.sparse_dense_matmul(norm_adj_mat, h), W), b)


def sigmoid(x):

    return 1.0/(1.0+np.exp(-x))