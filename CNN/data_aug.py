import tensorflow as tf
import numpy as np
from tensorflow import keras

train_datagen = ImageDataGenerator(
	rescale = 1./255,
	rotation_range = 40,
	width_shift_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True,
	fill_mode = 'nearest')