import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt

# Loading the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(train_images.shape)

# plt.imshow(train_images[0])
# plt.show()

model = keras.Sequential([
	keras.layers.Flatten(input_shape = (28, 28)),
	keras.layers.Dense(128, activation = tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)
	])

print(model.summary())

train_images = train_images / 255.0
test_images = test_images / 255.0
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 10)

print(model.evaluate(test_images, test_labels))

