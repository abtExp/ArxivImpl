import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
import cv2

import os

import face_recognition

import math

from os import listdir


def create_sketch():
	return sketch

# TODO : Create Hair Mask
def get_hair_mask():
	return np.zeros((256, 256), dtype=np.int32)

# Based On The Algorithm Mentioned In The Paper (Algorithm 1)
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

				mask = cv2.line(mask, (start_x, start_y), (end_x, end_y), (255, 255, 255), 10)

				start_x = end_x
				start_y = end_y

		mask = np.array(mask, dtype='int32')
		mask_panel[bbox[3]:bbox[1], bbox[0]:bbox[2]] = mask[:,:]

		# if np.random.randint(0, 10) > 5:
		# 	hair_mask = get_hair_mask()
		# 	mask += hair_mask

		return mask_panel

	else:
		return


def data_loader(vars):
	all_files = listdir(vars.DATA_INPUT_PATH)
	inps = []
	ops = []

	for file in all_files:
		image = cv2.imread(vars.DATA_INPUT_PATH+file)
		image = cv2.resize(image, vars.INP_SHAPE)

		mask = create_mask(image)

		mask = np.expand_dims(mask, axis=-1)

		reversed_mask = np.logical_not(mask).astype(np.int32)

		incomplete_image = np.multiply(reversed_mask, image)

		random_noise = np.zeros((np.shape(image)[0], np.shape(image)[1], 1))
		random_noise = cv2.randn(random_noise, 0, 255)
		random_noise = np.asarray(random_noise/255, dtype=np.uint8)

		img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
		sketch = cv2.Canny(img, 200, 200)
		sketch = np.expand_dims(sketch, axis=-1)
		sketch = np.multiply(mask, sketch)

		color = np.multiply(image, mask)

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
		ops.append(image)

	return np.array(inps), np.array(ops)