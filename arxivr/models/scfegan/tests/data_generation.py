import sys

sys.path.insert(0, 'D:/exp_labs/arxivr/arxivr/models/scfegan/')

from utils.utils import *
import cv2
import numpy as np

img = cv2.imread('C:/Users/abtex/Desktop/me.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
generated = cv2.imread('C:/Users/abtex/Desktop/me2.jpg')
generated = cv2.resize(generated, (img.shape[1], img.shape[0]))
generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)

# Testing Mask Generation
mask = create_mask(img)
mask = np.expand_dims(mask, axis=-1)
cv2.imshow('mask', mask)
cv2.waitKey(0)


# # Testing Sketch Information
# sketch = get_sketch_info(img, mask)
# cv2.imshow('sketch', sketch)
# cv2.waitKey(0)

# # Color Info
# color = get_color_info(img, mask)
# cv2.imshow('color', color)
# cv2.waitKey(0)

completed = complete_imgs(img, mask, generated)
cv2.imshow('completed', cv2.cvtColor(completed.astype(np.uint8), cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
