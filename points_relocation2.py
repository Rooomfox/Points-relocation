import numpy as np
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
# import matplotlib.pyplot as plt


def neural_network(dataset):
	dataset['group0'] = list(np.random.permutation(dataset['group0']))
	dataset['group1'] = list(np.random.permutation(dataset['group1']))
	dataset['group2'] = list(np.random.permutation(dataset['group2']))

	num_g0 = int(0.9 * len(dataset['group0']))
	num_g1 = int(0.9 * len(dataset['group1']))
	num_g2 = int(0.9 * len(dataset['group2']))

	data_t = dataset['group0'][:num_g0]
	data_t += dataset['group1'][:num_g1]
	data_t += dataset['group2'][:num_g2]
	data_t = np.random.permutation(data_t)
	x_train = [(p[0], p[1]) for p in data_t[:]]
	y_train = [p[2] for p in data_t[:]]

	data_v = dataset['group0'][num_g0:]
	data_v += dataset['group1'][num_g1:]
	data_v += dataset['group2'][num_g2:]
	data_v = np.random.permutation(data_v)
	x_val = [(p[0], p[1]) for p in data_v[:]]
	y_val = [p[2] for p in data_v[:]]

	scalarX, scalarY = MinMaxScaler(), MinMaxScaler()
	y_train = np.array(y_train).reshape(-1, 1)
	scalarX.fit(x_train)
	x_train2 = scalarX.transform(x_train)

	y_val = np.array(y_val).reshape(-1, 1)
	x_val2 = scalarX.transform(x_val)

	model = models.Sequential()
	model.add(layers.Dense(512, activation='relu', input_shape=(2,)))
	model.add(layers.Dense(128, activation='relu'))
	model.add(layers.Dense(32, activation='relu'))
	model.add(layers.Dense(3, activation='softmax'))

	model.compile(optimizer='rmsprop',
				  loss='sparse_categorical_crossentropy',
				  metrics=['acc'])
	model.save('saved_model')

	history = model.fit(x_train2,
						y_train,
						epochs=30,
						batch_size=100,
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
	for x in range(500, 2300, 5):
		for y in range(-600, 600, 5):
			p =(0.01*x, 0.01*y)
			x_test.append(p)
	x_test2 = scalarX.transform(x_test)
	y_test = model.predict(x_test2)

	predictions = []
	for index, pred in enumerate(y_test):
		x = round(x_test[index][0], 3)
		y = round(x_test[index][1], 3)
		label = np.argmax(pred)
		predictions.append((label, x, y))

	return predictions, history