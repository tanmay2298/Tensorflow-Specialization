import tensorflow as tf
tf.enable_eager_execution()

dataset = tf.data.Dataset.range(10)
dataset = dataset.window(5, shift = 1, drop_remainder = True)
dataset = dataset.flat_map(lambda window : window.batch(5))
dataset = dataset.map(lambda window : (window[:-1], window[-1:]))
dataset = dataset.shuffle(buffer_size = 10)

# dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

for x, y in dataset:
	print(x.numpy(), y.numpy())


model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(10, input_shape = [5], activation = 'relu'),
	tf.keras.layers.Dense(10, activation = 'relu'),
	tf.keras.layers.Dense(1)
	])

print(model.summary())
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
	lambda epoch : 1e-8 * 10 ** (epoch / 20))

optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9)
model.compile(loss = "mse", optimizer = optimizer)
model.fit(dataset, epochs = 100, verbose = 0, steps_per_epoch = 1)
