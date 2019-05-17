import numpy as np

import cv2

from os import listdir
from os.path import join
from PIL import Image, ImageOps

from lxml import etree

import matplotlib.pyplot as plt

import kmeans
import enet_utils

import gc

import keras

from sklearn.utils import shuffle

from matplotlib.path import Path
import matplotlib.patches as patches

from keras.layers import Conv2D, Dense, Input, BatchNormalization, Flatten, Dropout
from keras.models import Model

# def mAP(y_true, y_pred):
#     # for all queries
#     APS = []
#     for i in range(len(queries)):
#         precisions, recalls = queries[i]
#         max_recall = 1
#         max_precision = 0

#         for i in range(len(precisions)-1, 0, -1):
#             max_precision = np.maximum(precisions[i-1], max_precision)
#             precisions[i-1] = max_precision

#         index = np.where(recalls[1:] != recalls[:-1])[0]

#         AP = np.sum((recalls[i+1] - recalls[i]) * precisions[i + 1])

#         APS.append(AP)

#     mAP = np.sum(APS)/len(APS)

#     return mAP

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

def generate_mask(num_edges, pert, bbox, image):
	# pts = num_edges * 3 + 1
	# angles = np.linspace(0, 2*np.pi, pts)
	# codes = np.full(pts, Path.CURVE4)
	# codes[0] = Path.MOVETO

	# verts = np.stack((np.cos(angles)*0.5, np.sin(angles)*0.5)).T *(2*pert*np.random.random(pts)+1-pert)[:, None]
	# verts[-1, :] = verts[0, :]
	# path = Path(verts, codes)

	# fig = plt.figure(figsize=(bbox[2]/96, bbox[3]/96))
	# ax = fig.add_subplot(111)
	# ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
	# ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)

	# patch = patches.PathPatch(path, facecolor='none', transform=ax.transData)

	if type(image) == 'str':
		image = Image.open(join('../data/detector/train/logos/',image))
		# image = Image.open(join('/logos',image))

	# image = image.resize((bbox[2], bbox[3]))
	# image = np.array(image)
	# ax.axis('off')
	# im = ax.imshow(image)
	# ax.add_patch(patch)
	# im.set_clip_path(patch)

	# fig.canvas.draw()

	# out = np.array(fig.canvas.renderer._renderer)
	# out = cv2.resize(out, (bbox[2], bbox[3]))
	image = image.convert("RGBA")
	datas = image.getdata()

	newData = []
	for item in datas:
		if item[0] == 255 and item[1] == 255 and item[2] == 255:
			newData.append(((255, 255, 255, 0)))

		else:
			newData.append(item)

	image.putdata(newData)

	image = image.resize((bbox[2], bbox[3]))
	image = np.array(image)
	out = np.transpose(image, (1, 0, 2))

	# plt.close()

	# return only RGB
	return out

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

def load_data(directory, batch_size=10, img_dim=(512, 512), grid_dim=(16,16), boxes_per_cell=5, num_classes=194, anchors=[], logo_class_dict={}, max_boxes=5, mode='mask', labels_files_format='txt', vars = {}):

	flikr_class_logos_dict = vars.FLIKR_NUM_LOGO_IDS

	label_files = [file for file in listdir(directory) if file.endswith('.{}'.format(labels_files_format))]


	# img_files = [file for file in listdir(directory) if file.index('.') == len(file)-4 and file.endswith('png')]

	file_idxs = np.random.randint(0, len(label_files), batch_size)

	label_files = [label_files[i] for i in file_idxs]

	if mode == 'mask':
		all_mask_files = [file for file in listdir(directory) if file.endswith('png') and 'mask' in file]
		all_mask_files = [j for i in label_files for j in all_mask_files if i[:i.index('.')] in j]

	masks = []
	images = []
	labels = []

	for i in range(len(label_files)):

		if mode == 'mask':
			masks_for_this = [file for file in all_mask_files if label_files[i][:label_files[i].index('.')] in file]
			mask_label = np.zeros((img_dim[0], img_dim[1], num_classes))

		classes = np.zeros((num_classes))
		label = np.zeros((grid_dim[0], grid_dim[1], boxes_per_cell, 5+num_classes))

		# img = Image.open(join(directory, img_files[i])).resize((img_dim[0], img_dim[1]), Image.ANTIALIAS)
		# img = np.array(img)
		# images.append(np.array(img))

		grid_cell_size = img_dim[0]/grid_dim[0]

		data = []

		if labels_files_format == 'txt':
			with open(join(directory, label_files[i])) as f:
				data = f.read()
				data = data.split('\n')
				data = [i for i in data if len(i) > 0]

		elif labels_files_format == 'xml':
			xml = open(join(directory, label_files[i]))
			file = xml.read()
			tree = etree.fromstring(file)
			rect = {}

			for item in tree.iter('xmin', 'xmax', 'ymin', 'ymax', 'name'):
				if(item.tag in ['xmin','ymin','xmax','ymax']):
					rect[item.tag] = int(item.text)

				if(item.tag == 'name'):
					rect[item.tag] = int(logo_class_dict[item.text])

				if(len(rect) == 5):
					data.append(rect)
					rect = {}

			data = list(data.values())

		opened = False
		for j in range(len(data)):
			true_box_index = 0
			best_iou = anchors[true_box_index]
			box_data = data[j].split(' ')
			box_coords = box_data[0:4]
			box_coords = [int(i) for i in box_coords]
			box_class = int(box_data[4])

			if box_class in flikr_class_logos_dict.keys() and not opened:
				images.append(np.array(Image.open(join(directory, '{}.png'.format(label_files[i][:label_files[i].index('.')]))).resize((img_dim[0], img_dim[1]), Image.ANTIALIAS)))
				opened = True

			classes[box_class] = 1

			center_x = ((int(box_coords[0]) + int(box_coords[2]))*0.5)/img_dim[0]
			center_y = ((int(box_coords[1]) + int(box_coords[3]))*0.5)/img_dim[1]
			box_h = (int(box_coords[2]) - int(box_coords[0]))/img_dim[0]
			box_w = (int(box_coords[3]) - int(box_coords[1]))/img_dim[1]

			grid_row = int(center_x/grid_cell_size)
			grid_col = int(center_y/grid_cell_size)

			box = [1, center_x, center_y, box_w, box_h]

			for i in range(len(anchors)):
				curr_iou = iou([0, 0]+anchors[i], [0, 0]+box[3:])
				if curr_iou < best_iou:
					best_iou = curr_iou
					true_box_index = i

			label[grid_row, grid_col, true_box_index, :] = [*box, *classes]

			# For converting brand_text and brand_symbol into the same class
			box_class_name = flikr_class_logos_dict[box_class]
			box_class_name = box_class_name if '_' not in box_class_name else box_class_name[:box_class_name.index('_')]

			box_class = logo_class_dict[box_class_name.lower()]

			if mode == 'mask':
				mask_i = np.array(Image.open(join(directory,masks_for_this[j])).resize((img_dim[0], img_dim[1]), Image.ANTIALIAS))/255
				for l in range(mask_i.shape[0]):
					for k in range(mask_i.shape[1]):
						if mask_label[l,k,box_class] == 0:
							mask_label[l,k,box_class] = mask_i[l][k]

		combined_mask = generate_combined_mask(mask_label)
		last_mask = np.zeros((img_dim[0], img_dim[1]))
		for r in range(img_dim[0]):
			for c in range(img_dim[1]):
				if combined_mask[r,c] == 0:
					last_mask[r,c] = 1

		mask_label[:,:,-1] = last_mask[:,:]

		labels.append(label)
		if mode == 'mask':
			mask_label = np.reshape(mask_label, (img_dim[0]*img_dim[1], num_classes))
			masks.append(mask_label)

	# Leave it here to deal with grayscale
	images = np.reshape(images, (len(images), img_dim[0], img_dim[1], img_dim[2]))

	return np.array(images), np.array(labels), np.array(masks)


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

def bbox_from_mask(masks):
	boxes = []

	for i in range(np.shape(masks)[-1]):
		mask = masks[:,:,i]

		horizontal_idxs = np.where(np.any(mask, axis=0))[0]
		vertical_idxs = np.where(np.any(mask, axis=1))[0]

		if horizontal_idxs.shape[0]:
			x1, x2 = horizontal_idxs[[0,-1]]
			y1, y2 = vertical_idxs[[0, -1]]
			x2 = x2 + 1
			y2 = y2 + 1

		else:
			x1, x2, y1, y2 = 0,0,0,0

		boxes.append([x1, y1, x2, y2])

	return np.array(boxes)

def get_best_anchor_boxes(train_path, num_clusters, mode='txt'):

	grid_h = 1024 / 32
	grid_w = 1024 / 32

	labeled_images = [example for example in listdir(train_path) if example.endswith('.{}'.format(mode))]
	rectangles = []


	if mode == 'xml':
		for file in labeled_images:
			xml = open(join(train_path, file))
			file = xml.read()
			tree = etree.fromstring(file)
			rect = {}
			for item in tree.iter('xmin', 'xmax', 'ymin', 'ymax', 'name'):
				if item.tag in ['xmin', 'xmax', 'ymin', 'ymax']:
					rect[item.tag] = int(item.text)

				if(len(rect) == 5):
					rectangles.append(rect)
					rect = {}

		rectangles = [[float((rect['xmax']-rect['xmin'])/grid_h), float((rect['ymax']-rect['ymin'])/grid_w)] for rect in rectangles]

	elif mode == 'txt':
		for i in range(len(labeled_images)):
			with open(join(train_path, labeled_images[i])) as f:
				data = f.read()
				data = data.split('\n')
				for i in data:
					if len(i) > 0:
						box_data = i.split(' ')
						box_coords = box_data[0:4]
						box_h = float((int(box_coords[2]) - int(box_coords[0]))/grid_h)
						box_w = float((int(box_coords[3]) - int(box_coords[1]))/grid_w)
						rectangles.append([box_w, box_h])
				del data
				gc.collect()
	else:
		print('Invalid Mode.')

	best_anchors_ar, best_anchors = kmeans.kmeans(rectangles, num_clusters)

	return best_anchors_ar, best_anchors

# generate new training data for synthetic learning with random patches of logos placed inside random bboxes
def synthesize(train_path, inp_shape=(512,512,3), grid_shape=(16,16), num_classes=194, num_boxes=5, anchors=[], max_pts=5, pert=0.4, logo_class_dict={}, num_images=0, background_dir='./flikr/non_logo', backgrounds=None, logos=None, background_idxs=None, logo_idxs=None):
	images = []
	labels = []
	mask_labels = []
	true_boxes = []
	boxes = []

	nb_anchors = len(anchors)//2
	anchors = np.reshape(anchors, (nb_anchors, 2))

	background_dir = join(train_path, background_dir)
	logo_dir = join(train_path, './logos')

	# logo_dir = '/logos'

	all_logos = [i for i in listdir(logo_dir) if i.endswith('jpg') and i[:i.index('.')] in logo_class_dict.keys()]

	# logo_class from substring
	logo_classes = [int(logo_class_dict[i[:i.index('.')]]) for i in all_logos]

	if backgrounds == None:
		if background_idxs == None:
			# getting random backgrounds for synthesizing images
			background_idxs = np.random.rand((num_images))*len(listdir(background_dir))
			background_idxs = [int(a) for a in background_idxs]

		backgrounds = [Image.open(join(background_dir, i)).resize(tuple(inp_shape[0:2])) for i in [listdir(background_dir)[int(j)] for j in background_idxs]]

	if logos == None:
		if logo_idxs == None:
			# getting random logos for synthesizing images
			logo_idxs = np.random.rand((num_images))*len(all_logos)
			logo_idxs = [int(a) for a in logo_idxs]

		logos = [Image.open(join(logo_dir,j)) for j in [all_logos[k] for k in logo_idxs]]

	grid_cell_size = inp_shape[0]/grid_shape[0]

	for i, image in enumerate(backgrounds):
		# generating random bounding boxes
		# image = np.array(image)
		shape = np.shape(image)
		bbox = [
			np.random.randint(50,shape[0]-50),
			np.random.randint(50,shape[0]-50),
			np.random.randint(20, 100),
			np.random.randint(20, 100)
			]

		true_box_index = 0
		best_iou = 10000

		for anc in range(len(anchors)):
			gt_box = np.zeros((4,))
			boxx = np.zeros((4,))
			gt_box[2:] = anchors[anc]
			boxx[2:] = bbox[2:]
			curr_iou = iou(gt_box, boxx)
			if curr_iou < best_iou:
				best_iou = curr_iou
				true_box_index = anc

		mask = np.zeros((inp_shape[0], inp_shape[1], num_classes))

		logo = logos[i].resize((bbox[2], bbox[3]), Image.ANTIALIAS)

		logo = generate_mask(max_pts, pert, bbox, logo)
		# logo_activ = cv2.cvtColor(logo, cv2.COLOR_RGB2GRAY)
		# logo_activ = logo_activ/255
		# print('THE LOGO')
		# plt.imshow(logo_activ, cmap='gray')
		# plt.show()
		w=0
		h=0

		image = image.convert('RGBA')
		image = np.array(image)

		for y in range(bbox[0] - int(bbox[2]*0.5), (bbox[0] + int(bbox[2]*0.5))-1, 1):
			w = 0
			for x in range(bbox[1] - int(bbox[3]*0.5), (bbox[1] + int(bbox[3]*0.5))-1, 1):
				if logo[h,w,3] != 0:
					image[x,y,:] = logo[h,w,:]
					mask[x,y,logo_classes[i]] = 255
				w = w+1
			h = h+1

		last_mask = np.zeros((inp_shape[0], inp_shape[1]))
		for r in range(inp_shape[0]):
			for c in range(inp_shape[1]):
				if mask[r,c,logo_classes[i]] == 0:
					last_mask[r,c] = 255

		mask[:,:,-1] = last_mask

		center_x = ((int(bbox[0]) + int(bbox[2]))*0.5)/grid_cell_size
		center_y = ((int(bbox[1]) + int(bbox[3]))*0.5)/grid_cell_size
		box_h = (int(bbox[2]) - int(bbox[0]))/grid_cell_size
		box_w = (int(bbox[3]) - int(bbox[1]))/grid_cell_size

		grid_row = int(center_x)
		grid_col = int(center_y)

		image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

		image = np.array(image)
		# image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
		# image = np.array(image)
		images.append(image)

		img_class = np.zeros((num_classes))

		img_class[logo_classes[i]] = 1

		boxes.append([*bbox, logo_classes[i]])

		label = [1, *bbox, *img_class]

		boxes_labels = np.zeros((grid_shape[0], grid_shape[1],
								num_boxes, 1+4+num_classes))

		tr_bxes = np.zeros((1, 1, 1, 10, 4))

		if grid_row < grid_shape[0] and grid_col < grid_shape[1]:
			boxes_labels[int(grid_row), int(grid_col), true_box_index] = label

		tr_bxes[0, 0, 0, 0, :] = bbox

		labels.append(boxes_labels)
		mask = np.reshape(mask, (inp_shape[0]*inp_shape[1], num_classes))
		mask_labels.append(mask)
		true_boxes.append(tr_bxes)
	images = np.reshape(images, (len(images), inp_shape[0], inp_shape[1], inp_shape[2]))
	return np.array(images), np.array(labels), np.array(mask_labels), np.array(true_boxes), np.array(boxes)

def eval(model, vars):
	imgs = listdir(vars.EVAL_DICT)
	idxs = np.random.randint(0, len(imgs), vars.EVAL_BATCH_SIZE)
	eval_x = np.array([np.array(Image.open(join(vars.EVAL_DICT, imgs[i])).resize((vars.INP_SHAPE[0], vars.INP_SHAPE[1]))) for i in idxs])

	masks, preds, scores, scores_per_class = model.predict(eval_x)

	return masks, preds, scores, scores_per_class

class BootStrapGenerator(keras.utils.Sequence):
	def __init__(self, original_batch_size, synthetic_batch_size, vars, mode='train', out_mode='mask'):
		self.inp_shape = vars.INP_SHAPE
		self.vars = vars
		self.labels_files_format = vars.DETECTOR_LABEL_FILES_FORMAT
		self.original_bach_size = original_batch_size
		self.synthetic_batch_size = synthetic_batch_size
		self.batch_size = original_batch_size + synthetic_batch_size
		self.mode = mode
		self.out_mode = out_mode

	def __getitem__(self, index):
		x, box, y = [], [], []


		if self.out_mode == 'mask':
			x, y = self.__data_generation(None)
			index = np.random.randint(0, len(x))
			x = x[index]
			y = y[index]

			x = np.expand_dims(x, axis=0)
			y = np.expand_dims(y, axis=0)

			return x, y
		else:
			x, box, y = self.__data_generation(None)
			index = np.random.randint(0, len(x))
			box = box[index]
			box = np.expand_dims(box, axis=0)
			x = x[index]
			y = y[index]

			x = np.expand_dims(x, axis=0)
			y = np.expand_dims(y, axis=0)

			return [x, box], y

	def __len__(self):
		return self.vars.DETECTOR_BATCH_SIZE

	def __data_generation(self, a):
		directory = '{}/flikr/{}'.format(self.vars.DETECTOR_TRAIN_DATA_PATH, self.mode)
		original_x, original_y, original_mask, original_true_boxes, _ = load_valid_data(directory, self.original_bach_size, vars=self.vars)

		if self.mode == 'train':
			synthetic_x, synthetic_y, synthetic_mask, synthetic_true_boxes, _ = synthesize(
														self.vars.DETECTOR_TRAIN_DATA_PATH,
														self.vars.INP_SHAPE,
														(self.vars.DETECTOR_GRID_W, self.vars.DETECTOR_GRID_H),
														self.vars.LOGO_NUM_CLASSES,
														self.vars.DETECTOR_MAX_ANCHORS,
														self.vars.BEST_ANCHORS,
														max_pts=self.vars.DETECTOR_SYNTHETIC_MASK_MAX_POINTS,
														pert=self.vars.DETECTOR_SYNTHETIC_MASK_PERT,
														logo_class_dict=self.vars.FLIKR_ONLY_LOGO_CLASS_DICT,
														num_images = self.synthetic_batch_size
												)
			train_x = np.array([*original_x, *synthetic_x])
			train_y = np.array([*original_y, *synthetic_y])
			train_mask = np.array([*original_mask, *synthetic_mask])
			train_true_boxes = np.array([*original_true_boxes, *synthetic_true_boxes])

		else:
			train_x = np.array(original_x)
			train_y = np.array(original_y)
			train_mask = np.array(original_mask)
			train_true_boxes = np.array(original_true_boxes)

		if self.out_mode == 'mask':
			return train_x, train_mask
		else:
			return np.array(train_x), np.array(train_true_boxes), np.array(train_y)

def yolo_target_from_enet(masks, vars, anchors):
	labels = []
	for i in range(len(masks)):
		label = np.zeros((vars.DETECTOR_GRID_H, vars.DETECTOR_GRID_W, vars.MAX_BOXES_PER_CELL, 1+4+vars.LOGO_NUM_CLASSES))
		classes = np.zeros((vars.LOGO_NUM_CLASSES))
		for mask in masks[i]:
			mask = np.reshape(mask, (vars.INP_SHAPE[0], vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
			for j,m in enumerate(mask):
				bbox = bbox_from_mask(m)

				classes[j] = 1

				center_x = ((int(bbox[0]) + int(bbox[2]))*0.5)/vars.INP_SHAPE[0]
				center_y = ((int(bbox[1]) + int(bbox[3]))*0.5)/vars.INP_SHAPE[1]
				box_h = (int(bbox[2]) - int(bbox[0]))/vars.INP_SHAPE[0]
				box_w = (int(bbox[3]) - int(bbox[1]))/vars.INP_SHAPE[1]

				grid_row = int(center_x/vars.DETECTOR_GRID_H)
				grid_col = int(center_y/vars.DETECTOR_GRID_W)

				box = [1, center_x, center_y, box_w, box_h]

				for k in range(len(anchors)):
					curr_iou = anchors[k] - (box_h * box_w)
					if curr_iou < best_iou:
						best_iou = curr_iou
						true_box_index = k

				label[grid_row, grid_col, true_box_index, :] = [*box, *classes]
			labels.append(label)


	return labels

def apply_random_alpha(image):
	image = np.array(image)
	np.random.seed(100)
	alpha = np.random.randint(170, 255)
	image[:,:,-1] = alpha
	return Image.fromarray(image)

def apply_random_crop(image):
	shape = np.shape(image)
	random_patch_dims = (
			np.random.randint(0, 10), #x0
			np.random.randint(0, 10), #y0
			np.random.randint(shape[0]-20, shape[0]), #x1
			np.random.randint(shape[1]-20, shape[1])  #y1
		)

	image = image.crop(random_patch_dims)

	image = image.resize((shape[0], shape[1]))

	return image

def randomizer(batch_size, logo_class_dict, vars={}):
	path = './data/detector/train/flikr/train'
	train_x = []
	train_y = []

	flikr_class_logos_dict = vars.FLIKR_NUM_LOGO_IDS
	rotations = np.arange(0,360,15)

	image_files = [file[:file.index('.')] for file in listdir(path) if file.index('.') == len(file)-4 and file.endswith('png')]
	indices = np.random.randint(0, len(image_files), batch_size)
	image_files = [image_files[i] for i in indices]

	for file in image_files:
		image = Image.open('{}/{}.png'.format(path,file))
		with open('{}/{}.gt_data.txt'.format(path, file)) as f:
			bboxes = f.read()
			bboxes = bboxes.split('\n')
			bboxes = [i for i in bboxes if len(i) > 0]
			bboxes = [i.split(' ') for i in bboxes]

		for bbox in bboxes:
			box_coords = [int(index) for index in bbox[0:4]]
			box_class = int(bbox[4])
			box_class_name = flikr_class_logos_dict[box_class]
			box_class_name = box_class_name if '_' not in box_class_name else box_class_name[:box_class_name.index('_')]

			box_class = logo_class_dict[box_class_name.lower()]
			img = image.crop(box_coords)
			# img = img.convert('LA')
			# img = apply_random_alpha(img)
			# img = apply_random_crop(img)
			angles = np.random.choice(rotations, 3, replace=False)
			indices = [i for j in angles for i in range(len(rotations)) if j == rotations[i]]
			for i, angle in enumerate(angles):
				img = img.rotate(angle)
				img = img.resize((64,64), Image.ANTIALIAS)
				label = np.zeros(24)
				label[indices[i]] = 1
				# cv2.imwrite('./imgs/imgs/{}.{}.{}.png'.format(box_class_name, file, angle), np.array(img))
				train_x.append(np.array(img))
				train_y.append(label)
	return shuffle(np.array(train_x), np.array(train_y))

def logo_loader(logo_dict, batch_size):
	train_x = []
	train_y = []
	path = './data/detector/train/logos'
	images = [i for i in listdir(path) if i.endswith('jpg') and i[:i.index('.')] in logo_dict.keys()]

	idxs = np.random.randint(0, len(images), batch_size)
	images = [images[i] for i in idxs]

	for image in images:
		img = Image.open(join(path, image))
		img = generate_mask(None, None, [0,0,64,64], img)
		img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
		train_x.append(np.array(img))
		# label = np.zeros(194)
		label = logo_dict[image[:image.index('.')].lower()]
		train_y.append(label)

	return shuffle(np.array(train_x), np.array(train_y))
class GeneratorLoader(keras.utils.Sequence):
	def __init__(self, vars, mode='enc'):
		self.mode = mode
		self.vars = vars
		self.batch_size = self.vars.ENC_BATCH_SIZE

	def __getitem__(self, index):
		x, y = self.__data_generation(None)

		index = np.random.randint(0, self.batch_size)

		x = x[index]
		y = y[index]

		x = np.reshape(x, (1, np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]))
		y = np.reshape(y, (1, np.shape(y)[0]))

		return x, y

	def __len__(self):
		return len(listdir('./data/detector/train/logos'))

	def __data_generation(self, a):
		if self.mode == 'enc':
			return logo_loader(self.vars.FLIKR_ONLY_LOGO_CLASS_DICT, self.batch_size)

		elif self.mode == 'or':
			return randomizer(self.batch_size, logo_class_dict=self.vars.FLIKR_ONLY_LOGO_CLASS_DICT, vars=self.vars)

class EvaluationCallback(keras.callbacks.Callback):
	def __init__(self, name, inp_shape, formatter, plotter, class_dict):
		self.name = name
		self.inp_shape = inp_shape
		self.formatter = formatter
		self.plotter = plotter
		self.class_dict = class_dict
		super(EvaluationCallback, self).__init__()

	def on_epoch_end(self, epoch, logs=None):
		eval_dir = './data/detector/test/logo+/'
		images = listdir(eval_dir)
		image_idx = np.random.randint(0, len(images))
		image = Image.open(join(eval_dir, images[image_idx])).resize((self.inp_shape[0], self.inp_shape[1]))
		image = np.array(image)
		cv2.imwrite('./test/test_{}/img.{}.png'.format(self.name, epoch), image)
		image = np.expand_dims(image, axis=0)

		if self.name == 'enet':
			masks, preds, scores, scores_per_class = self.formatter(self.model.predict(image))
			res = self.plotter(epoch, preds, scores_per_class, self.color_dict)
			res = np.expand_dims(res, axis=-1)
		elif self.name == 'yolo':
			true_boxes = np.zeros((1, 1, 1, 1, 10, 4))
			boxes, confs, coords, classes = self.formatter(self.model.predict([image, true_boxes]))
			res = self.plotter(epoch, image, coords, classes, confs, self.class_dict)

		cv2.imwrite('./test/test_{}/img.{}.res.png'.format(self.name, epoch), res)

def gen_loader(batch_size, vars):
	directory = './data/generator/train'

	files = listdir(directory)
	idxs = np.random.randint(0, len(files), batch_size)

	images = [np.array(Image.open(join(directory, files[i])).resize((vars.GEN_INP_SHAPE[0], vars.GEN_INP_SHAPE[1]), Image.ANTIALIAS)) for i in idxs]
	class_labels = [int(files[i][:files[i].index('.')]) for i in idxs]

	return np.array(images), np.array(class_labels)

def read_data(directory, batch_size=128, mode='txt', vars={}):
	logo_class_dict = {'pepsi': 0, 'stellaartois': 1, 'becks': 2, 'guinness': 3, 'bmw': 4, 'singha': 5, 'esso': 6, 'texaco': 7, 'fosters': 8, 'fedex': 9, 'corona': 10, 'erdinger': 11, 'paulaner': 12, 'ford': 13, 'adidas': 14, 'heineken': 15, 'chimay': 16, 'nvidia': 17, 'dhl': 18, 'shell': 19, 'starbucks': 20, 'ferrari': 21, 'carlsberg': 22, 'cocacola': 23, 'hp': 24, 'ups': 25, 'tsingtao': 26, 'milka': 27, 'rittersport': 28, 'apple': 29, 'aldi': 30, 'google': 31, '__empty__': 32}
	flikr_class_num_dict = {0: 'HP', 1: 'adidas_symbol', 2: 'adidas_text', 3: 'aldi', 4: 'apple', 5: 'becks_symbol', 6: 'becks_text', 7: 'bmw', 8: 'carlsberg_symbol', 9: 'carlsberg_text', 10: 'chimay_symbol', 11: 'chimay_text', 12: 'cocacola', 13: 'corona_symbol', 14: 'corona_text', 15: 'dhl', 16: 'erdinger_symbol', 17: 'erdinger_text', 18: 'esso_symbol', 19: 'esso_text', 20: 'fedex', 21: 'ferrari', 22: 'ford', 23: 'fosters_symbol', 24: 'fosters_text', 25: 'google', 26: 'guinness_symbol', 27: 'guinness_text', 28: 'heineken', 29: 'milka', 30: 'nvidia_symbol', 31: 'nvidia_text', 32: 'paulaner_symbol', 33: 'paulaner_text', 34: 'pepsi_symbol', 35: 'pepsi_text', 36: 'rittersport', 37: 'shell', 38: 'singha_symbol', 39: 'singha_text', 40: 'starbucks', 41: 'stellaartois_symbol', 42: 'stellaartois_text', 43: 'texaco', 44: 'tsingtao_symbol', 45: 'tsingtao_text', 46: 'ups'}

	detected_masks = []
	labels = []
	masked_images = []
	bbox_coords = []
	images = []

	all_files = listdir(directory)

	all_files = np.array([f for f in all_files if f.endswith('.txt')])

	idxs = np.random.randint(0, len(all_files), batch_size)

	print(len(idxs))

	files = [all_files[i] for i in idxs]

	if mode == 'txt':
		for file in files:
			img_name = '{}.png'.format(file[:file.index('.')])
			img = Image.open(join(directory, img_name))
			images.append(np.array(img))
			with open(join(directory, file)) as f:
				data = f.read()
				bboxes = data.split('\n')
				bboxes = [bbox for bbox in bboxes if len(bbox) > 0]
				for bbox in bboxes:
					curr_img = img
					box_coords = bbox.split(' ')[0:4]
					box_coords = [int(coord) for coord in box_coords]
					class_id = int(bbox.split(' ')[4])
					class_id = flikr_class_num_dict[class_id].lower()
					class_id = class_id if '_' not in class_id else class_id[:class_id.index('_')]
					class_id = logo_class_dict[class_id]
					detected = np.array(curr_img.crop(box_coords).resize((64, 64), Image.ANTIALIAS))
					masked_img = np.array(curr_img)
					masked_img[box_coords[1]:box_coords[3], box_coords[0]:box_coords[2], :] = 0
					masked_img = cv2.resize(masked_img, (vars.INP_SHAPE[0], vars.INP_SHAPE[1]))
					detected_masks.append(detected)
					labels.append(class_id)
					masked_images.append(masked_img)
					bbox_coords.append(box_coords)
					if len(masked_images) == batch_size:
						return np.array(masked_images), np.array(labels), np.array(detected_masks), np.array(bbox_coords), np.array(images)

	# else:
	#     while len(masked_images) < batch_size:
	#         idx = np.random.randint(0, len(all_files))
	#         img = Image.open(join(directory, all_files[idx][0]))
	#         images.append(np.array(img))
	#         for bbox in all_files[idx][1]:
	#             bbox = [int(i) for i in bbox]
	#             curr_img = np.array(img)
	#             xmin = int(bbox[0] - 0.5 * bbox[2])
	#             xmax = int(bbox[0] + 0.5 * bbox[2])
	#             ymin = int(bbox[1] - 0.5 * bbox[3])
	#             ymax = int(bbox[1] + 0.5 * bbox[3])

	#             bbox = [xmin, ymin, xmax, ymax]

	#             detected = cv2.resize(curr_img[ymin:ymax, xmin:xmax, :], (64, 64))
	#             class_id = all_files[idx][2].lower()
	#             has__ = True if '_' in class_id else False
	#             if has__:
	#                 class_id = flikr_class_num_dict[class_id]
	#                 class_id = class_id if '_' not in class_id else class_id[:class_id.index('_')]
	#             class_id = logo_class_dict[class_id]
	#             masked_img = np.array(curr_img)
	#             masked_img[bbox[1]-32:bbox[1]+32, bbox[0]-32:bbox[0]+32, :] = 0
	#             masked_img = cv2.resize(masked_img, (vars.INP_SHAPE[0], vars.INP_SHAPE[1]))
	#             detected_masks.append(detected)
	#             labels.append(class_id)
	#             masked_images.append(masked_img)
	#             bbox_coords.append(bbox)

	return np.array(masked_images), np.array(labels), np.array(detected_masks), np.array(bbox_coords), np.array(images)

def ccngan_data_loader(batch_size, vars):
	from scipy.io import loadmat

	dir1 = './data/detector/train/flikr/train'
	dir2 = './data/detector/train/flikr/valid'
	# dir3 = './data/detector/test/logo+'

	# matdir = './data/detector/train/groundtruth.mat'

	# all_flikr_d1 = np.array([file for file in listdir(dir1) if file.endswith('.txt')])
	# all_flikr_d2 = np.array([file for file in listdir(dir2) if file.endswith('.txt')])
	# all_logo_plus_data = loadmat(matdir)

	print(batch_size)
	random_batch_size = batch_size // 2
	print(random_batch_size)
	# remaining = (batch_size - (2*random_batch_size))

	# logo_32_plus = []

	# for file in all_logo_plus_data['groundtruth'][0]:
	#     curr_img_data = []
	#     curr_file_name = file[0][0][file[0][0].rindex('\\')+1:]
	#     curr_bboxes = file[1]
	#     curr_class = file[2][0]

	#     curr_img_data.append(curr_file_name)
	#     curr_img_data.append(cxurr_bboxes)
	#     curr_img_data.append(curr_class)

	#     logo_32_plus.append(curr_img_data)

	# all_logo_plus_data = np.array(logo_32_plus)

	all_data_imgs, all_data_labels, all_data_patches, all_data_box_coords = [], [], [], []

	if batch_size != 1:
		d1, ld1, pd1, bc1, _ = read_data(dir1, batch_size=random_batch_size, vars=vars)
		d2, ld2, pd2, bc2, _ = read_data(dir2, batch_size=random_batch_size, vars=vars)
		# d3, ld3, pd3, bc3, _ = read_data(dir3, all_logo_plus_data, remaining, (len(d1)+len(d2)), 'mat', vars=vars)

		all_data_imgs = np.concatenate((d1, d2))
		all_data_labels = np.concatenate((ld1, ld2))
		all_data_patches = np.concatenate((pd1, pd2))
		all_data_box_coords = np.concatenate((bc1, bc2))
	else:
		all_data_imgs, all_data_labels, all_data_patches, all_data_box_coords = read_data(dir1, batch_size=batch_size, vars=vars)
	return all_data_imgs, all_data_labels, all_data_patches, all_data_box_coords

def get_orientation_encoder():
	inp = Input(shape=(64, 64, 3))
	x = Conv2D(filters=16, kernel_size=(1,1), strides=(2,2), activation='relu')(inp)
	x = BatchNormalization()(x)
	x = Conv2D(filters=16, kernel_size=(2,2), strides=(2,2), activation='relu')(x)
	x = Flatten()(x)
	x = Dense(units=1024, activation='tanh')(x)
	x = Dropout(0.3)(x)
	x = Dense(units=512, activation='tanh')(x)
	x = Dense(units=24, activation='softmax')(x)

	model = Model(inputs=inp, outputs=x)

	return model

def orientation_encoder(frame, boxes, vars):
	if not vars.LOADED_ORIENTATION_ENCODER:
		encoder = get_orientation_encoder(vars)
		check_dir = vars.OR_CHECKPOINTS_PATH
		encoder.load_weights(check_dir, listdir(check_dir)[-1])
		vars.ORIENTATION_ENCODER = encoder
		vars.LOADED_ORIENTATION_ENCODER = True
	else:
		encoder = vars.ORIENTATION_ENCODER

	orientations = []

	for box in boxes:
		orientation = encoder.predict(cv2.resize(frame[box[1]:box[3], box[0]:box[2]], vars.GEN_INP_SHAPE))
		orientations.append(orientation)

	return orientations, vars