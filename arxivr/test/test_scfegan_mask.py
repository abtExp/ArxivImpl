import sys
sys.path.insert(0, './utils')

from scfegan_utils import create_mask

import numpy as np
import cv2

image = cv2.imread('F:/celeba-dataset/img_align_celeba/000001.jpg')
img = cv2.resize(image, (256, 256))
mask = create_mask(img)
mask = np.expand_dims(mask, axis=-1)
reversed_mask = np.logical_not(mask).astype(np.int32)
imgg = np.multiply(img, reversed_mask)
cv2.imwrite('./face.jpg', imgg)