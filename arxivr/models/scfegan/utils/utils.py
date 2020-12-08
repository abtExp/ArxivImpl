import os
import cv2
import math
import numpy as np
from os import listdir
import face_recognition
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from .face_toolbox_keras.models.detector import face_detector
from .face_toolbox_keras.models.parser.face_parser import FaceParser


'''
	Extract sketch information using edge detection

	image : input image
	mask : binary mask

	returns :
				sketch : sketch information
'''
def get_sketch_info(image, mask):
	sketch = cv2.Canny(image, 200, 200)
	sketch = np.expand_dims(sketch, axis=-1)
	sketch = np.multiply(mask, sketch)
	return sketch


'''
	Returns Mask For Hair

	image : input image

	returns  :
				component_mask : hair mask
'''
def get_hair_mask(image):
	parser = FaceParser()
	img = image[..., ::-1]
	parsed = parser.parse_face(img, with_detection=False)
	component_mask = np.zeros(tuple(image.shape[:-1]))
	component_mask[parsed[0] == 17] = 1

	return component_mask


'''
	Get the average color of sections to create the color information

	image : the input image
	mask : binary mask

	returns :
				color : color information for the input
'''
def get_color_info(image, mask):
	annotation_colors = [
		'0, background', '1, skin', '2, left eyebrow', '3, right eyebrow',
		'4, left eye', '5, right eye', '6, glasses', '7, left ear', '8, right ear', '9, earings',
		'10, nose', '11, mouth', '12, upper lip', '13, lower lip',
		'14, neck', '15, neck_l', '16, cloth', '17, hair', '18, hat'
	]

	parser = FaceParser()
	img = image[..., ::-1]
	parsed = parser.parse_face(img, with_detection=False)

	# blurring image
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	imgg = cv2.medianBlur(img, 5)
	bilateral = imgg
	for i in range(20):
		bilateral = cv2.bilateralFilter(bilateral, 20, 30, 30)

	# Converting to the median color based on the segmentation and then applying mask
	median_image = np.zeros(image.shape)
	for i in range(len(annotation_colors)):
		component_mask = np.zeros(tuple(image.shape[:-1]))
		component_mask[parsed[0] == i] = 1
		masked = np.multiply(cv2.cvtColor(bilateral, cv2.COLOR_RGB2BGR), np.expand_dims(component_mask, axis=-1))
		median_image += masked

	color = np.multiply(median_image, mask)

	return color


'''
	Create Masks Based On The Algorithm Mentioned In The Paper (Algorithm 1)

	img : ground truth input image
	max_draws : number of line draws
	max_len : maximum length of a line
	max_angle : maximum angle of a line
	max_lines : maximum number of lines
	shape : input shape of the image

	returns :
				mask_panel : binary mask

'''
def create_mask(img, max_draws=10, max_len=50, max_angle=60, max_lines=10, shape=(256, 256)):
	mask_panel = np.zeros(shape)

	num_lines = np.random.randint(0, max_draws)

	bbox = face_recognition.face_locations(img)[0]

	if len(bbox) > 0:
		mask = np.zeros((bbox[1]-bbox[3], bbox[2]-bbox[0]))

		for i in range(0, num_lines):
			start_x = np.random.randint(0, bbox[2]-bbox[0])
			start_y = np.random.randint(0, bbox[1]-bbox[3])
			start_angle = np.random.randint(0, 360)
			num_vertices = np.random.randint(0, max_lines)

			for j in range(0, num_vertices):
				angle_change = np.random.randint(-max_angle, max_angle)
				if j%2 == 0:
					angle = start_angle + angle_change
				else:
					angle = start_angle + angle_change + 180

				length = np.random.randint(0, max_len)

				end_x = start_x+int(length * math.cos(math.radians(angle)))
				end_y = start_y+int(length * math.sin(math.radians(angle)))

				mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), (255, 255, 255), 5)

				start_x = end_x
				start_y = end_y

		mask = np.array(mask, dtype='int32')
		mask_panel[bbox[3]:bbox[1], bbox[0]:bbox[2]] = mask[:,:]

		if np.random.randint(0, 10) > 5:
			hair_mask = get_hair_mask(img)
			mask += hair_mask

		return mask_panel
	else:
		return


'''
	Complete images by replacing the non-removed part of the generated images by ground truth image

	images : ground truth image
	masks : binary mask for image
	generated : output of the generator

	returns :
				completed_images : generated image with the non-removed part replaced by ground truth image
'''
def complete_imgs(images, masks, generated):
    patches = generated * masks
    reversed_mask = np.logical_not(masks, dtype=np.int)

    completion = images * reversed_mask
    completed_images = np.add(patches, completion)

    return completed_images


'''
	Data loader

	config : configuration object

	returns :
				inps : input to the generator : Input image (RGB) (Incomplete)
												sketch information
												color information
												random noise
												binary mask
				ops : ground truth image
'''
def data_loader(config, gen=False):
	all_files = listdir(config.DATA_INPUT_PATH)
	inps = []
	ops = []

	for file in all_files:
		image = cv2.imread(config.DATA_INPUT_PATH+file)
		image = cv2.resize(image, config.INP_SHAPE)

		mask = create_mask(image)

		mask = np.expand_dims(mask, axis=-1)

		reversed_mask = np.logical_not(mask).astype(np.int32)

		incomplete_image = np.multiply(reversed_mask, image)

		if gen:
			op = np.zeros(8*8*256)
		else:
			op = np.ones(8*8*256)

		random_noise = np.zeros((np.shape(image)[0], np.shape(image)[1], 1))
		random_noise = cv2.randn(random_noise, 0, 255)
		random_noise = np.asarray(random_noise/255, dtype=np.uint8)

		sketch = get_sketch_info(image, mask)

		color = get_color_info(image, mask)

		inp = np.concatenate(
			[
				incomplete_image,
				mask,
				sketch,
				color,
				random_noise
			]
		, axis=-1)

		inps.append(inp)
		ops.append(op)

	return np.array(inps), np.array(ops)


'''
	Pass the input image through VGG-16 pretrained on ImageNet
	and get outputs from layers pool1, pool2, and pool3

	feature_extractor : VGG-16 model instance
	x : input image

	returns :
			outputs : Activations of layers as a list
			nf : Number of features in each layer as a list
'''
def extract_features(feature_extractor, x):
	nf = []
	functors = []

	for layer in feature_extractor.layers:
		if layer.name in ["block1_pool", "block2_pool", "block3_pool"]:
			nf.append(np.prod(layer.output_shape[1:]))
			functors.append(layer.output)

	appl = K.function([feature_extractor.input, K.learning_phase()], functors)
	outputs = appl([x, 1.])
	outputs = [out[0] for out in outputs]

	return outputs, nf


class RandomWeightedAverage(Layer):
	"""Provides a (random) weighted average between real and generated image samples"""
	def call(self, inputs):
		alpha = K.random_uniform(inputs[0].shape)
		return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])