import tensorflow as tf
import numpy as np
from tensorflow import keras

# Loading the data
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

model = keras.models.Sequential([
	keras.layers.Conv2D(64, (3, 3), activation = 'relu', input_shape = (28, 28, 1)),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Flatten(),
	keras.layers.Dense(128, activation = 'relu'),
	keras.layers.Dense(10, activation = 'softmax')
	])

print(model.summary())

train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)

train_images = train_images / 255.0
test_images = test_images / 255.0

model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model.fit(train_images, train_labels, epochs = 10)

print(model.evaluate(test_images, test_labels))
