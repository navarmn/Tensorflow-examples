import os
FILEPATH = os.path.dirname(__file__)

import tensorflow as tf
from sklearn.datasets import fetch_california_housing
import numpy as np


housing = fetch_california_housing()
m, n  = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m,1)), housing.data]


X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')
Xt = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(Xt, X)), Xt), y)


with tf.Session() as sess:
    with tf.summary.FileWriter(os.path.join(FILEPATH, 'summaries'), sess.graph) as writer:
        print("Calculation Theta")
        print(theta.eval())
        