import tensorflow
import numpy as np
from tensorflow import keras
from keras.optimizers import RMSprop
model = keras.models.Sequential([
	keras.layers.Conv2D(16, (3, 3), activation = 'relu', input_shape = (150, 150, 3)),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(32, (3, 3), activation = 'relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Conv2D(64, (3, 3), activation = 'relu'),
	keras.layers.MaxPooling2D(2, 2),
	keras.layers.Flatten(),
	keras.layers.Dense(512, activation = 'relu'),
	keras.layers.Dense(1, activation = 'sigmoid')
])

print(model.summary())

model.compile(loss = 'binary_crossentropy',
	optimizer = RMSprop(lr = 0.001),
	metrics = ['acc'])

history = model.fit_generator(
	train_generator,
	steps_per_epoch = 8,
	epochs = 15,
	validation_data = validation_generator,
	validation_steps = 8,
	verbose = 2)
	)