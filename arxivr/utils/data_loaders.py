import keras
from keras.utils import Sequence

from PIL import Image
import numpy as np

from os import listdir, path
from os.path import join

import sys
sys.path.insert(0, './utils')

from scfegan_utils import *

import cv2

def compute_overlap(c1, c2):
	x1, x2 = c1
	x3, x4 = c2

	if x3 < x1:
		if x4 < x1:
			return 0
		else:
			return min(x2, x4) - x1
	else:
		if x2 < x3:
			return 0
		else:
			return min(x2, x4) - x3

def iou(box1, box2):
	int_w, int_h = 0, 0
	intersection = 0
	union = 0

	int_w = compute_overlap([box1[0], box1[2]], [box2[0], box2[2]])
	int_h = compute_overlap([box1[1], box1[3]], [box2[1], box2[3]])

	intersection = int_h * int_w

	w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
	w2, h2 = box2[2] - box2[0], box2[3] - box2[1]

	union = w1*h1 + w2*h2 - intersection

	iou_ = float(intersection)/union
	return iou_
class BBOX():
	def __init__(self, coords, c=None, classes=None):
		self.xmin, self.ymin, self.xmax, self.ymax = coords
		self.c = c
		self.classes = classes
		self.label = -1
		self.score = -1
		self.box_coords = coords

	def get_label(self):
		self.label = np.argmax(self.classes) if self.label == -1 else self.label
		return self.label

	def get_score(self):
		self.score = self.classes[self.get_label()] if self.score == -1 else self.score
		return self.score

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


def enet_data_loader(vars, mode):
	batch_size = vars.TRAIN_BATCH_SIZE
	images, _, masks, _, _ = load_valid_data(vars.SCFEGAN_DATA_INPUT_PATH, batch_size, vars=vars)
	return images, masks


# def scfegan_data_loader(vars, mode='train', mask_mode='create'):
# 	# if mask_mode == create, use create_mask on every image

# 	batch_size = vars.SCFEGAN_BATCH_SIZE
# 	images, _, masks, _, boxes = load_valid_data(vars.SCFEGAN_DATA_INPUT_PATH, batch_size, vars=vars)

# 	# Removing the void class mask

# 	if mask_mode == 'detected':
# 		masks = np.reshape(masks, (np.shape(masks)[0], vars.INP_SHAPE[0], vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
# 		reversed_masks = masks[:,:,:,-1]
# 		masks = masks[:,:,:,:-1]

# 		detected_masks = [generate_combined_mask(masks[i]) for i in range(np.shape(masks)[0])]

# 	inputs = []
# 	all_images = []

# 	for i, image in enumerate(images):
# 		input_image = np.array(image)
# 		cv2.imwrite('./input.jpg', input_image)
# 		random_noise = np.zeros((np.shape(image)[0], np.shape(image)[1], 1))
# 		random_noise = cv2.randn(random_noise, 0, 255)
# 		random_noise = np.asarray(random_noise/255, dtype=np.uint8)

# 		# Normalizing Input Image
# 		input_image = cv2.resize(input_image, (vars.INP_SHAPE[0], vars.INP_SHAPE[1]))
# 		input_image = (input_image / 127.5) - 1.

# 		if mask_mode == 'detected':
# 			detected_mask = detected_masks[i]
# 			reversed_mask = reversed_masks[i]
# 		else:
# 			detected_mask = np.array(create_mask()/255, dtype=np.int32)
# 			reversed_mask = np.logical_not(detected_mask).astype(np.int32)

# 		cv2.imwrite('./mask.jpg', detected_mask*255)
# 		cv2.imwrite('./reversed_mask.jpg', reversed_mask*255)

# 		img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# 		sketch = cv2.Canny(img, 200, 200)
# 		sketch = np.multiply(detected_mask, sketch)

# 		cv2.imwrite('./sketch.jpg', sketch)

# 		reversed_mask = np.expand_dims(reversed_mask, axis=-1)
# 		detected_mask = np.expand_dims(detected_mask, axis=-1)
# 		sketch = np.expand_dims(sketch, axis=-1)

# 		input_image = np.multiply(reversed_mask, input_image)
# 		random_noise = np.multiply(detected_mask, random_noise)

# 		color = np.multiply(image, detected_mask)
# 		cv2.imwrite('./color.jpg', color)
# 		cv2.imwrite('./noise.jpg', random_noise)

# 		inp = np.concatenate(
# 			[
# 				input_image,
# 				detected_mask,
# 				sketch,
# 				color,
# 				random_noise
# 			]
# 		, axis=-1)

# 		inputs.append(inp)
# 		all_images.append(image)

# 	return np.array(inputs), np.array(all_images)


def get_sketch(mask):
	sketch = None
	return sketch

class DATA_LOADER(keras.utils.Sequence):
	def __init__(self, vars, mode, loader, args=None):
		self.vars = vars
		self.mode = mode
		self.loader = loader
		self.args = args

	def __getitem__(self, index):
		x, y = self.__data_generation([])
		index = np.random.randint(0, len(x))

		x, y = x[index], y[index]

		x = np.expand_dims(x, axis=0)
		y = np.expand_dims(y, axis=0)

		return x, y

	def __len__(self):
		return 100

	def __data_generation(self, l):
		if self.args:
			x, y = self.loader(self.vars, self.mode, self.args)
		else:
			x, y = self.loader(self.vars, self.mode)

		return x, y


class ENET_DATA_LOADER(DATA_LOADER):
	def __init__(self, vars, mode):
		super(ENET_DATA_LOADER, self).__init__(vars, mode, enet_data_loader)


class SCFEGAN_DATA_LOADER(DATA_LOADER):
	def __init__(self, vars, mode):
		super(SCFEGAN_DATA_LOADER, self).__init__(vars, mode, scfegan_data_loader)