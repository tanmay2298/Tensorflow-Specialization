import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

model = tf.keras.Sequential([
	tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64), return_sequences = True),
	tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
	tf.keras.layers.Dense(64, activation = 'relu'),
	tf.keras.layers.Dense(1, activation = 'sigmoid')
])

print(model.summary())