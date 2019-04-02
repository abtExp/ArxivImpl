from keras.layers import Layer, Conv2D, Conv2DTranspose, Reshape, Multiply, LeakyReLU, Activation, multiply, BatchNormalization
from keras.initializers import RandomNormal
import keras.backend as K
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import os

import sys
sys.path.append(0, './settings')

import vars

from os import listdir

from utils import utils

class LRNLayer(Layer):
	def __init__(self, alpha=1e-4, beta=0.75, k=2, n=5):
		super(LRNLayer, self).__init__()
		self.alpha = alpha
		self.beta = beta
		self.k = k
		self.n = n

	def call(self, x):
		op = []
		nc = np.shape(x)[-1]
		for i in range(nc):
			sq = K.sum((x[:,:,:,max(0, int(i-self.n/2)):min(nc-1, i+int(self.n/2))+1]) ** 2)
			op.append(x[:,:,:,i]/((self.k + self.alpha * sq) ** self.beta))

		op = tf.convert_to_tensor(op)

		op = tf.transpose(op, perm=[1,2,3,0])

		op_shape = self.compute_output_shape(np.shape(x))

		op._keras_shape = op_shape

		return op

	def compute_output_shape(self, input_shape):
		return input_shape

	def compute_mask(self, input, input_mask):
		return 1*[None]

class GatedDeConv(Layer):
	def __init__(self, out_shape, kernel_size, strides, std_dev):
		super(GatedDeConv, self).__init__()
		self.out_shape = out_shape
		self.kernel_size = kernel_size
		self.strides = strides
		self.std_dev = std_dev

	def call(self, x):
		inp = x

		kernel = K.random_uniform_variable(shape=(self.kernel_size[0], self.kernel_size[1], self.out_shape[-1], int(x.get_shape()[-1])), low=0, high=1)

		deconv = K.conv2d_transpose(x, kernel=kernel, strides=self.strides, output_shape=self.out_shape, padding='same')

		biases = K.zeros(shape=(self.out_shape[-1]))

		deconv = K.reshape(K.bias_add(deconv, biases), deconv.get_shape())
		deconv = LeakyReLU()(deconv)

		g = K.conv2d_transpose(inp, kernel, output_shape=self.out_shape, strides=self.strides, padding='same')
		biases2 = K.zeros(shape=(self.out_shape[-1]))
		g = K.reshape(K.bias_add(g, biases2), deconv.get_shape())

		g = K.sigmoid(g)

		deconv = tf.multiply(deconv, g)

		outputs = [deconv, g]

		output_shapes = self.compute_output_shape(x.shape)
		for output, shape in zip(outputs, output_shapes):
			output._keras_shape = shape

		return [deconv, g]

	def compute_output_shape(self, input_shape):
		return [self.out_shape, self.out_shape]

	def compute_mask(self, input, input_mask=None):
		return 2 * [None]


# Load everything
def load_valid_data(dir, batch_size, format='txt', mode='mask', vars={}, debug=False):
	flikr_possible_images = vars.FLIKR_NUM_LOGO_IDS

	nb_anchors = len(vars.BEST_ANCHORS)//2
	anchors = [BBOX((0, 0, vars.BEST_ANCHORS[2*i], vars.BEST_ANCHORS[2*i+1])) for i in range(int(len(vars.BEST_ANCHORS)//2))]

	all_label_files = [file for file in listdir(dir) if file.endswith(format)]
	all_img_files = [image for image in listdir(dir) if image.endswith('.png')]
	all_mask_files = [mask for mask in listdir(dir) if 'mask' in mask]

	labels = []
	images = []
	true_boxes = []
	masks = []
	boxes = []
	true_images = []

	grid_cell_size = float(vars.INP_SHAPE[0]/vars.DETECTOR_GRID_H)

	while len(images) < batch_size:
		boxes_for_this = []
		label_names = []
		idx = np.random.randint(0, len(all_label_files))
		file = all_label_files[idx]

		with open(join(dir, file)) as f:
			data = f.read()
			data = data.split('\n')
			data = [i for i in data if len(i) > 0]

		if int(data[0].split(' ')[4]) in flikr_possible_images.keys():
			image = Image.open(join(dir, '{}.png'.format(file[:file.index('.')])))
			images.append(np.array(image.resize((vars.INP_SHAPE[0], vars.INP_SHAPE[1]), Image.ANTIALIAS)))

			label = np.zeros((vars.DETECTOR_GRID_H, vars.DETECTOR_GRID_W, vars.DETECTOR_MAX_ANCHORS, 1+4+vars.LOGO_NUM_CLASSES))
			true_box = np.zeros((1, 1, 1, vars.DETECTOR_MAX_DETECTIONS_PER_IMAGE, 4))
			true_box_idx = 0

			if mode == 'mask':
				mask = np.zeros((vars.INP_SHAPE[0], vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
				masks_for_this = [mask for mask in all_mask_files if file[:file.index('.')] in mask]

			for j in range(len(data)):
				classes = np.zeros((vars.LOGO_NUM_CLASSES))

				best_anchor_idx = 0
				best_iou = 10000
				box_data = data[j].split(' ')
				box_coords = box_data[0:4]
				box_coords = [int(i) for i in box_coords]
				box_class = int(box_data[4])

				box_class_name = flikr_possible_images[box_class]
				box_class_name = box_class_name if '_' not in box_class_name else box_class_name[:box_class_name.index('_')]

				box_class = vars.FLIKR_ONLY_LOGO_CLASS_DICT[box_class_name.lower()]
				label_names.append(box_class_name.lower())
				classes[box_class] = 1

				xmin, ymin, xmax, ymax = box_coords

				xmin = int((xmin * vars.INP_SHAPE[0])/np.shape(image)[1])
				xmax = int((xmax * vars.INP_SHAPE[0])/np.shape(image)[1])

				ymin = int((ymin * vars.INP_SHAPE[1])/np.shape(image)[0])
				ymax = int((ymax * vars.INP_SHAPE[1])/np.shape(image)[0])

				xmin = max(min(xmin, vars.INP_SHAPE[0]), 0)
				xmax = max(min(xmax, vars.INP_SHAPE[0]), 0)

				ymin = max(min(ymin, vars.INP_SHAPE[1]), 0)
				ymax = max(min(ymax, vars.INP_SHAPE[1]), 0)

				box_coords = [xmin, ymin, xmax, ymax]

				center_x = ((box_coords[0] + box_coords[2])*0.5)
				center_y = ((box_coords[1] + box_coords[3])*0.5)

				grid_row = int(np.floor(center_x)/grid_cell_size)
				grid_col = int(np.floor(center_y)/grid_cell_size)

				if grid_row < vars.DETECTOR_GRID_H and grid_col < vars.DETECTOR_GRID_W:
					box_h = (box_coords[2] - box_coords[0])
					box_w = (box_coords[3] - box_coords[1])

					box = [1, center_x, center_y, box_w, box_h]
					for i in range(nb_anchors):
						gt_box = np.zeros((4,))
						boxx = np.zeros((4,))
						gt_box[2:] = anchors[i].box_coords[2:]
						boxx[2:] = box[3:]
						curr_iou = iou(gt_box, boxx)
						if curr_iou < best_iou:
							best_iou = curr_iou
							best_anchor_idx = i

					box = [int(i) for i in box]

					boxes_for_this.append([*box[1:], box_class])

					label[grid_row, grid_col, best_anchor_idx, :] = [*box, *classes]
					true_box[0,0,0,true_box_idx,:] = box[1:]

					true_box_idx += 1
					true_box_idx = true_box_idx % vars.DETECTOR_MAX_DETECTIONS_PER_IMAGE

				mask_i = np.array(Image.open(join(dir, masks_for_this[j])).resize((vars.INP_SHAPE[0], vars.INP_SHAPE[1]), Image.ANTIALIAS))/255

				for l in range(mask_i.shape[0]):
					for k in range(mask_i.shape[1]):
						if mask[l,k,box_class] == 0:
							mask[l,k,box_class] = mask_i[l][k]

			combined_mask = generate_combined_mask(mask)

			last_mask = np.zeros((vars.INP_SHAPE[0], vars.INP_SHAPE[1]))
			for r in range(vars.INP_SHAPE[0]):
				for c in range(vars.INP_SHAPE[1]):
					if combined_mask[r,c] == 0:
						last_mask[r,c] = 1

			mask[:,:,-1] = last_mask[:,:]
			mask = np.reshape(mask, (vars.INP_SHAPE[0]*vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
			masks.append(mask)
			labels.append(label)
			true_boxes.append(true_box)
			boxes.append(boxes_for_this)

			if debug:
				for box in boxes_for_this:
					imgg = images[-1]
					xmin = int((box[0] - (box[3] * 0.5)))
					xmax = int((box[0] + (box[3] * 0.5)))
					ymin = int((box[1] - (box[2] * 0.5)))
					ymax = int((box[1] + (box[2] * 0.5)))
					cv2.rectangle(imgg, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
					cv2.putText(
								imgg, label_names[-1],
								(int(box[0]), int(box[1])),
								0, 1.2e-3 * np.shape(imgg)[0],
								(0, 255, 0), 2
							)
					cv2.imshow('img', imgg)
					cv2.waitKey(0)

	return np.array(images), np.array(labels), np.array(masks), np.array(true_boxes), np.array(boxes)


def generate_combined_mask(masks, color_code_dict = {}):
	shape = (masks.shape[0], masks.shape[1])
	img = np.zeros(shape)
	masks = masks.transpose()
	masks = np.transpose(masks, axes=(0,2,1))
	masks = masks.reshape((masks.shape[0], masks.shape[1]*masks.shape[2]))
	masks = masks/255

	img = np.reshape(img,[-1])

	for i in range(masks.shape[0]):
		for j in range(masks.shape[1]):
			if img[j] == 0:
				img[j] = masks[i][j]


	img = np.reshape(img, shape)

	return img


# Generate Input Data For SCFEGAN From Flikr47 dataset
def gen_inp(vars, dir=''):
	batch_size = vars.SCFEGAN_BATCH_SIZE
	images, _, masks, _, boxes = load_valid_data('./data/detector/train/flikr/train/', batch_size, vars=vars)
	logo_dir = './data/detector/train/logos'

	# Removing the void class mask
	masks = np.reshape(masks, (np.shape(masks)[0], vars.INP_SHAPE[0], vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
	reversed_masks = masks[:,:,:,-1]
	masks = masks[:,:,:,:-1]

	detected_masks = [utils.generate_combined_mask(masks[i]) for i in range(np.shape(masks)[0])]

	inputs = []
	all_images = []

	for i, image in enumerate(images):
		input_image = np.array(image)
		random_noise = np.zeros((np.shape(image)[0], np.shape(image)[1], 1))
		random_noise = cv2.randn(random_noise, 0, 255)
		random_noise = np.asarray(random_noise/255, dtype=np.uint8)

		# Normalizing Input Image
		input_image = cv2.resize(input_image, (vars.INP_SHAPE[0], vars.INP_SHAPE[1]))
		input_image = (input_image / 127.5) - 1.

		detected_mask = detected_masks[i]
		reversed_mask = reversed_masks[i]

		img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sketch = cv2.Canny(img, 100, 100)
		sketch = np.multiply(detected_mask, sketch)

		reversed_mask = np.expand_dims(reversed_mask, axis=-1)
		detected_mask = np.expand_dims(detected_mask, axis=-1)
		sketch = np.expand_dims(sketch, axis=-1)

		input_image = np.multiply(reversed_mask, input_image)
		random_noise = np.multiply(detected_mask, random_noise)

		color = np.multiply(image, detected_mask)

		# input_image = np.expand_dims(input_image, axis=0)
		# random_noise = np.expand_dims(random_noise, axis=0)
		# sketch = np.expand_dims(sketch, axis=0)
		# color = np.expand_dims(color, axis=0)
		# detected_mask = np.expand_dims(detected_mask, axis=0)

		inp = np.concatenate(
			[
				input_image,
				detected_mask,
				sketch,
				color,
				random_noise
			]
		, axis=-1)

		inputs.append(inp)
		all_images.append(image)

	return np.array(inputs), np.array(all_images)
