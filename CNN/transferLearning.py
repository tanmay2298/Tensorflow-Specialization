import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dense(1, activation = 'sigmoid')(x)


model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr = 0.0001),
			loss = 'binary_crossentropy',
			metrics = ['acc'])