import matplotlib.pyplot as plt
# import numpy as np
import csv
import math
import time

from points_relocation_3d2 import neural_network


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
	num_l_p = int(0.8 * total)
	num_v_p = int(0.2 * total)

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


def add_label(*groups, model='2d', angle=None):
	result = {}
	if model == '2d':
		for label, points in enumerate(groups):
			group = f'group{label}'
			result[group] = []
			for j, _ in enumerate(points[0]):
				x = round(points[0][j], 2)
				y = round(points[1][j], 2)
				label2 = label
				feature = (x, y, label2)
				result[group] += (feature, )
	elif model == '3d':
		for label, points in enumerate(groups):
			group = f'group{label}'
			result[group] = []
			for j, _ in enumerate(points[0]):
				x = points[0][j] * math.cos(angle)
				y = points[0][j] * math.sin(angle)
				z = points[1][j]
				r = points[0][j]
				theta = math.degrees(angle)
				x = round(x, 2)
				y = round(y, 2)
				z = round(z, 2)
				r = round(r, 2)
				theta = round(theta, 2)
				feature = (x, y, z, r, theta, label)
				# feature = (x, y, z, label)
				result[group] += (feature, )
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

		x_array0, y_array0, z_array0 = [], [], []
		x_array1, y_array1, z_array1 = [], [], []
		x_array2, y_array2, z_array2 = [], [], []

		for r_p in relocated_points:
			if r_p[0] == 0:
				x_array0.append(r_p[1])
				y_array0.append(r_p[2])
				z_array0.append(r_p[3])
			elif r_p[0] == 1:
				x_array1.append(r_p[1])
				y_array1.append(r_p[2])
				z_array1.append(r_p[3])
			else:
				x_array2.append(r_p[1])
				y_array2.append(r_p[2])
				z_array2.append(r_p[3])

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.set_xlim(0, 25)
		ax.set_ylim(0, 25)
		ax.set_zlim(-7, 7)

		ax.scatter(x_array0, y_array0, z_array0, alpha = 1, color = 'r', s = 2)
		ax.scatter(x_array1, y_array1, z_array1, alpha = 0.5, color = 'b', s = 2)
		ax.scatter(x_array2, y_array2, z_array2, alpha = 1, color = 'g', s = 2)
		plt.savefig(f'result/relocated_fig_{filename[-6:-4]}.jpg')
		plt.show()
		plt.clf()


def draw_loss(filename, loss, history):
	if loss == True:
		loss = history.history['loss']
		val_loss = history.history['val_loss']
		epochs = range(1, len(loss) + 1)
		print(loss)
		print(val_loss)

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
		print(acc)
		print(val_acc)

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
	start1 = time.time()
	results = {'group0':[],
			   'group1':[],
			   'group2':[]}
	for i in range(36):
		if i < 10:
			extend = f'0{i}'
		else:
			extend = i	
		filename2 = f'data/FFHR-d1_R15.6m_B4.7T_R3.50_g1.20_20181105_leg_poincare-{extend}.dat'
		filename1 = f'data3.29/LCFS1/FFHR-d1_R15.6m_B4.7T_R3.50_g1.20_20181105_LCFS_poincare-{extend}.dat'
		filename3 = f'outer_points/table{extend}.csv'
		try:
			inner_points = readfile(filename1, 41)
			wanted_points = readfile(filename2)
			with open(filename3, encoding='utf-8') as file_object:
				lines = file_object.readlines()
		except:
			pass
		else:
			x_array, y_array = [], []
			for index, line in enumerate(lines):
				if index > 1:
					line = line.strip().split(',')
					x = (2500 - int(line[0])) * 0.01
					y = (int(line[1]) - 700) * 0.01
					x_array.append(x)
					y_array.append(y)
			outer_points = [x_array, y_array]
			outer_points[0] = 2 * outer_points[0]
			outer_points[1] = 2 * outer_points[1]
			wanted_points[0] = 10 * wanted_points[0]
			wanted_points[1] = 10 * wanted_points[1]
			for j in range(3):
				angle = math.radians(36 * j + i)
				if angle <= 90:
					result = add_label(outer_points, wanted_points, inner_points,
									   model='3d', angle=angle)
					for key, value in result.items():
						results[key] += value
	time1 = time.time() - start1
	print(f'time of data preprocessing is {time1:.2f} s.')

	output_info(len(results['group0']),
				len(results['group1']),
				len(results['group2']))

	# print(add_label(outer_points, wanted_points, inner_points))

	start2 = time.time()
	predictions, history = neural_network(results)
	time2 = time.time() - start2
	print(f'time of compiling "points_relocation_3d2" is {time2:.2f} s.')

	controller(filename3,
			   loss = True,
			   acc = True,
		 	   original_fig = False,
			   relocated_fig = False,
			   create_file = True,
			   predictions = predictions,
			   history = history)