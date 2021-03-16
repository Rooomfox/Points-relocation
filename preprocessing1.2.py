import matplotlib.pyplot as plt
import csv

from points_relocation2 import neural_network
# from points_relocation2_grid_search import neural_network


def readfile(filename, index=None):
	with open(filename) as file_object:
		lines = file_object.readlines()

	Newlist = []
	for line in lines:
		Newlist.append(line.split())

	points = []
	if index:
		for line in Newlist:
			if int(line[0]) <= 41:
				x = float(line[2])
				y = float(line[3])
				point = (x, y)
				points.append(point)
	else:
		for line in Newlist:
			x = float(line[2])
			y = float(line[3])
			point = (x, y)
			points.append(point)
	points = list(zip(*points))
	return points


def output_csv_file(filename, *groups):
	t1 = filename[-6:]
	t2 = 'classification'
	filename2 = t2 + t1
	with open(filename2, 'w', encoding='utf-8', newline='') as file_object:
		writer = csv.writer(file_object)
		for label, points in enumerate(groups):
			for j, _ in enumerate(points[0]):
				x = f'{points[0][j]:.2f}'
				y = f'{points[1][j]:.2f}'
				label2 = str(label)
				writer.writerow((x,y,label2))
	return 'file has been made.'


def output_csv_file_2(filename, create_file, predictions):
	if create_file == True:
		t1 = filename[-6:]
		t2 = 'relocated_points_'
		filename2 = 'result/' + t2 + t1
		with open(filename2, 'w', encoding='utf-8', newline='') as file_object:
			writer = csv.writer(file_object)
			for point in predictions:
				writer.writerow(point)
	return 'file has been made.'


def output_info(*info):
	num_o_p = info[0]
	num_m_p = info[1]
	num_i_p = info[2]
	others = info[3:]
	total = num_o_p + num_m_p + num_i_p
	num_l_p = int(0.9 * total)
	num_v_p = int(0.1 * total)

	print(f'the number of outer points:{num_o_p}')
	print(f'the number of middle points:{num_m_p}')
	print(f'the number of inner points:{num_i_p}')

	filename = 'result/info.txt'
	with open(filename, 'w') as file_object:
		f = file_object
		f.write('The number of different kinds of points:\n')
		f.write(f'outer points:{num_o_p}\n')
		f.write(f'middle points:{num_m_p}\n')
		f.write(f'inner points:{num_i_p}\n')
		f.write(f'learning points:{num_l_p}\n')
		f.write(f'validation points:{num_v_p}\nend')


def add_label(*groups):
	result = {}
	for label, points in enumerate(groups):
		group = f'group{label}'
		result[group] = []
		for j, _ in enumerate(points[0]):
			x = round(points[0][j], 2)
			y = round(points[1][j], 2)
			label2 = label
			feature = (x, y, label2)
			result[group] += (feature,)
	return result


def draw_original_fig(filename, original_fig):
	if original_fig == True:
		plt.scatter(x_array, y_array, alpha = 1, color = 'r', s = 2)
		plt.scatter(inner_points[0], inner_points[1], alpha = 1, color = 'g', s = 2)
		plt.scatter(wanted_points[0], wanted_points[1], alpha = 1, color = 'b', s = 2)
		plt.axis([0, 25, -7, 7])
		plt.xlabel('x axis')
		plt.ylabel('y axis')
		plt.savefig(f'result/original_fig_{filename[-6:-4]}.jpg')
		plt.show()
		plt.clf()


def draw_relocated_fig(filename, relocated_fig, predictions):
	if relocated_fig == True:
		relocated_points = predictions

		x_array0, y_array0 = [], []
		x_array1, y_array1 = [], []
		x_array2, y_array2 = [], []

		for r_p in relocated_points:
			if r_p[0] == 0:
				x_array0.append(r_p[1])
				y_array0.append(r_p[2])
			elif r_p[0] == 1:
				x_array1.append(r_p[1])
				y_array1.append(r_p[2])
			else:
				x_array2.append(r_p[1])
				y_array2.append(r_p[2])

		plt.scatter(x_array0, y_array0, alpha = 1, color = 'r', s = 2)
		plt.scatter(x_array1, y_array1, alpha = 1, color = 'b', s = 2)
		plt.scatter(x_array2, y_array2, alpha = 1, color = 'g', s = 2)
		plt.axis([0, 25, -7, 7])
		plt.xlabel('x axis')
		plt.ylabel('y axis')
		plt.savefig(f'result/relocated_fig_{filename[-6:-4]}.jpg')
		plt.show()
		plt.clf()


def draw_loss(filename, loss, history):
	if loss == True:
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		epochs = range(1, len(loss) + 1)

		plt.plot(epochs, loss, 'bo', label='Training loss')
		plt.plot(epochs, val_loss, 'b', label='Validation loss')
		plt.title('Training and Validation loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()
		plt.savefig(f'result/loss_{filename[-6:-4]}.jpg')
		plt.show()
		plt.clf()


def draw_acc(filename, acc, history):
	if acc == True:
		acc = history.history['acc']
		val_acc = history.history['val_acc']
		epochs = range(1, len(acc) + 1)

		plt.plot(epochs, acc, 'bo', label='Training acc')
		plt.plot(epochs, val_acc, 'b', label='Validation acc')
		plt.title('Training and Validation accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()
		plt.savefig(f'result/acc_{filename[-6:-4]}.jpg')
		plt.show()
		plt.clf()


def controller(filename, loss, acc, original_fig, relocated_fig, 
				create_file, predictions, history):
	draw_loss(filename, loss, history)
	draw_acc(filename, acc, history)
	draw_original_fig(filename, original_fig)
	draw_relocated_fig(filename, relocated_fig, predictions)
	output_csv_file_2(filename, create_file, predictions)


if __name__ == '__main__':
	filename1 = 'data/FFHR-d1_R15.6m_B4.7T_R3.50_g1.20_20181105_leg_poincare-27.dat'
	filename2 = 'data3.29/LCFS1/FFHR-d1_R15.6m_B4.7T_R3.50_g1.20_20181105_LCFS_poincare-27.dat'
	filename3 = 'outer_points/table27.csv'
	with open(filename3, encoding='utf-8') as file_object:
		lines = file_object.readlines()

	x_array, y_array = [], []
	for index, line in enumerate(lines):
		if index > 1:
			line = line.strip().split(',')
			x = (2500 - int(line[0])) * 0.01
			y = (int(line[1]) - 700) * 0.01
			x_array.append(x)
			y_array.append(y)

	outer_points = [x_array, y_array]
	outer_points[0] = 7 * outer_points[0]
	outer_points[1] = 7 * outer_points[1]
	inner_points = readfile(filename2, 41)
	wanted_points = readfile(filename1)
	wanted_points[0] = 14 * wanted_points[0]
	wanted_points[1] = 14 * wanted_points[1]

	output_info(len(outer_points[0]),
				len(wanted_points[0]),
				len(inner_points[0]))
	# print(add_label(outer_points, wanted_points, inner_points))
	predictions, history = neural_network(add_label(
						   outer_points, wanted_points, inner_points))

	controller(filename3,
			   loss = True,
			   acc = True,
		 	   original_fig = True,
			   relocated_fig = True,
			   create_file = True,
			   predictions = predictions,
			   history = history)