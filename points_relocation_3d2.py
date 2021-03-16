import numpy as np
import math
from tensorflow.keras import models
from tensorflow.keras import layers
# from tensorflow.keras import regularizers
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt


def neural_network(dataset):
	dataset['group0'] = list(np.random.permutation(dataset['group0']))
	dataset['group1'] = list(np.random.permutation(dataset['group1']))
	dataset['group2'] = list(np.random.permutation(dataset['group2']))

	num_g0 = int(0.95 * len(dataset['group0']))
	num_g1 = int(0.95 * len(dataset['group1']))
	num_g2 = int(0.95 * len(dataset['group2']))

	data_t = dataset['group0'][:num_g0]
	data_t += dataset['group1'][:num_g1]
	data_t += dataset['group2'][:num_g2]
	data_t = np.random.permutation(data_t)
	x_train = [(p[0], p[1], p[2], p[3], p[4]) for p in data_t[:]]
	y_train = [p[5] for p in data_t[:]]
	# x_train = [(p[0], p[1], p[2]) for p in data_t[:]]
	# y_train = [p[3] for p in data_t[:]]

	data_v = dataset['group0'][num_g0:]
	data_v += dataset['group1'][num_g1:]
	data_v += dataset['group2'][num_g2:]
	data_v = np.random.permutation(data_v)
	x_val = [(p[0], p[1], p[2], p[3], p[4]) for p in data_v[:]]
	y_val = [p[5] for p in data_v[:]]
	# x_val = [(p[0], p[1], p[2]) for p in data_v[:]]
	# y_val = [p[3] for p in data_v[:]]

	scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
	y_train = np.array(y_train).reshape(-1, 1)
	scalarX.fit(x_train)
	x_train2 = scalarX.transform(x_train)

	y_val = np.array(y_val).reshape(-1, 1)
	x_val2 = scalarX.transform(x_val)

	model = models.Sequential()
	model.add(layers.Dense(1024, activation='relu', input_shape=(5,)))
	model.add(layers.Dense(256, activation='relu'))
	# model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(3, activation='softmax'))

	model.compile(optimizer='rmsprop',
				  loss='sparse_categorical_crossentropy',
				  metrics=['acc'])

	history = model.fit(x_train2,
						y_train,
						epochs=17,
						batch_size=1000,
						validation_data=(x_val2, y_val))

	loss = history.history['loss']
	val_loss = history.history['val_loss']

	epochs = range(1, len(loss) + 1)

	# plt.plot(epochs, loss, 'bo', label='Training loss')
	# plt.plot(epochs, val_loss, 'b', label='Validation loss')
	# plt.title('Training and Validation loss')
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.legend()

	acc = history.history['acc']
	val_acc = history.history['val_acc']

	# plt.plot(epochs, acc, 'bo', label='Training acc')
	# plt.plot(epochs, val_acc, 'b', label='Validation acc')
	# plt.title('Training and Validation accuracy')
	# plt.xlabel('Epochs')
	# plt.ylabel('Accuracy')
	# plt.legend()
	# plt.show()

	x_test = []
	for x in range(0, 240, 1):
		for y in range(0, 240, 1):
			for z in range(-70, 70, 1):
				p = (0.1*x, 0.1*y, 0.1*z)
				r = math.sqrt(p[0]**2 + p[1]**2)
				if r == 0:
					theta = 0
				else:
					theta = math.asin(p[1]/r)
					theta = math.degrees(theta)
					theta = round(theta, 2)
				p1 = (r, theta)
				p2 = p + p1
				x_test.append(p2)
				# x_test.append(p)
	x_test2 = scalarX.transform(x_test)
	y_test = model.predict(x_test2)

	predictions = []
	for index, pred in enumerate(y_test):
		x = round(x_test[index][0], 2)
		y = round(x_test[index][1], 2)
		z = round(x_test[index][2], 2)
		label = np.argmax(pred)
		predictions.append((label, x, y, z))

	return predictions, history