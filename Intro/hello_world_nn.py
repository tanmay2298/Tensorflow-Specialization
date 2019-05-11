import tensorflow as tf
from tensorflow import keras
import numpy as np

model = tf.keras.Sequential([tf.keras.layers.Dense(units = 1, input_shape = [1])])
model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error') #sgd - stochastic gradient descent

print(model.summary())

# Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype = float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype = float)

model.fit(xs, ys, epochs = 5000)

print(model.predict([10.0]))

for layer in model.layers:
	print("Layer : ", layer)
	weights = layer.get_weights()
	print(weights)