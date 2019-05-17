import sys
sys.path.insert(0, './utils')

from utils import load_valid_data

import numpy as np
import keras.backend as K
import matplotlib.pyplot as plt
from keras.callbacks import Callback

INP_SHAPE = (1024, 1024, 3)
LOGO_NUM_CLASSES = 33

def enet_outputs(preds, inp_shape=INP_SHAPE, num_classes=LOGO_NUM_CLASSES):
	masks = np.reshape(preds, (np.shape(preds)[0], inp_shape[0], inp_shape[1], num_classes))
	scores = np.max(masks, axis=-1)
	preds = np.argmax(masks, axis=-1)
	scores_per_class = [np.sum(scores[preds == c]) / np.sum(scores) for c in range(num_classes)]

	return [masks, preds, scores, scores_per_class]

def enet_loss(y_true, y_pred):
	weights = K.flatten(y_true)
	classes = np.bincount(weights)
	probs = classes/194
	weights = 1 / (np.log(1.02 + probs))
	# scale predictions so that the class probas of each sample sum to 1
	y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
	# clip to prevent NaN's and Inf's
	y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
	# calc
	loss = y_true * K.log(y_pred) * weights
	loss = -K.sum(loss, -1)
	return loss

# apply pixel classes depending upon the probability per class prediction
def get_preds_per_pixel(classes, probs, colors):
	print(np.shape(classes))
	image = np.zeros((np.shape(classes)[0], np.shape(classes)[1], 3))

	for r in range(np.shape(classes)[0]):
		for c in range(np.shape(classes)[1]):
			if probs[classes[r,c]] > 0.3:
				image[r,c,:] = colors[classes[r,c]][:]

	return image

def plot(epoch, predicted_classes, probs_per_class, color_dict, mode='bw', plt_mode = 'classif'):
	predicted_classes = predicted_classes[0]
	if plt_mode == 'probs':
		image = get_preds_per_pixel(predicted_classes, probs_per_class, color_dict)

	else:
		if mode == 'bw':
			nc = 1
		else:
			nc = 3
		image = np.zeros((np.shape(predicted_classes)[0], np.shape(predicted_classes)[0], nc))

		if mode == 'bw':
			image = predicted_classes
		else:
			for r in range(np.shape(predicted_classes)[0]):
				for c in range(np.shape(predicted_classes)[1]):
					image[r,c,:] = color_dict[predicted_classes[r,c]][:]

	# if mode == 'bw':
	# 	plt.imshow(image, cmap='gray')
	# else:
	# 	plt.imshow(image)

	# plt.show()
	return image

def data_loader(vars, mode):
	images, _, masks, _, _ = utils.load_valid_data(vars.SCFEGAN_DATA_INPUT_PATH, batch_size, vars=vars)
	return images, masks