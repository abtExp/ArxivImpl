from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as K

import tensorflow as tf

import cv2
import numpy as np

model = VGG16(input_shape=(512, 512, 3), include_top=False, weights='imagenet')

img = cv2.imread('C:/Users/abtex/Desktop/me.jpg')
img = cv2.resize(img, (512, 512))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.expand_dims(img, axis=0)

# cv2.imshow('img', img)
# cv2.waitKey(0)

outputs = []
nfs = []

functors = []

for layer in model.layers:
    if layer.name in ["block1_pool", "block2_pool", "block3_pool"]:
        functors.append(layer.output)
        nfs.append(np.prod(layer.output_shape[1:]))
        # outputs.append(tf.make_ndarray(layer.output))

funcs = K.function([model.input, K.learning_phase()], functors)
outputs = funcs([img, 1.])
outputs = [out[0] for out in outputs]

print(nfs)

for output in outputs:
    print(output.shape)