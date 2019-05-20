import tensorflow as tf
import numpy as np
from tensorflow import keras

train_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
train_dir,
target_size = (300, 300),
batch_size = 128,
class_mode = 'categorical') #instead of binary


model = keras.models.Sequential([
...
...

...

...
keras.layers.Dense(3, activation = 'softmax')]) # instead of 1 sigmoid neuron (3 - since there are 3 classes and softmax will select the highest probabilty)

from keras.optimizers import RMSprop

model.compile(loss = 'categorical_crossentropy',
optimizers = RMSprop(lr = 0.001),
metrics = ['acc']) #instead of binary_crossentropy