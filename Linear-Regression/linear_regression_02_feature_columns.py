import os
FILEPATH = os.path.dirname(__file__)

import tensorflow as tf
from tensorflow.estimator import LinearRegressor
from sklearn.datasets import fetch_california_housing
import numpy as np


housing = fetch_california_housing()
m, n  = housing.data.shape

# Dictonary values
# iterable = []
# for i in range(0,n):
#     # iterable.append(zip(housing.feature_names[i], housing.data[:,i]))
#     iterable.append((housing.feature_names[i], housing.data[:,i]))

# x_dict = {key: value for (key, value) in iterable}

x_dict = {
    'MedInc': housing.data[:,0],
    'HouseAge': housing.data[:,1],
    'AveRooms': housing.data[:,2],
    'AveBedrms': housing.data[:,3],
    'Population': housing.data[:,4],
    'AveOccup': housing.data[:,5],
    'Latitude': housing.data[:,6],
    'Longitude': housing.data[:,7]
}   

# Using the feature columns
feat_col = [tf.feature_column.numeric_column(k) for k in housing.feature_names]

def input_fn(X, y):
    return  tf.estimator.inputs.numpy_input_fn(
                x = X,
                y = y,
                num_epochs=2,
                shuffle=True,
                batch_size=256
            )

model = LinearRegressor(feat_col, os.path.join(FILEPATH, 'model_trained'))

# model.train(input_fn=input_fn(housing.data, housing.target.reshape((-1,1))))
model.train(input_fn=input_fn(x_dict, housing.target), steps=2)