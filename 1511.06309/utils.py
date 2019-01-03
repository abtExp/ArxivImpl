import tensorflow as tf
import numpy as np

def huber_loss(del_t, delta=1.0):
  cond  = tf.keras.backend.abs(del_t) < delta

  squared_loss = 0.5 * tf.keras.backend.square(del_t)
  linear_loss  = delta * (tf.keras.backend.abs(del_t) - 0.5 * delta)

  return tf.where(cond, squared_loss, linear_loss)


def grid_generator(h, w):
	grid = np.zeros((h, w, 3))
	# Create Grid Generator and Transform Matrix Generator Based On STN model
	return grid