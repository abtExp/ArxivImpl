import cv2
import numpy as np
import tensorflow as tf

def gp_loss(img, mask, discriminator_model):
    with tf.GradientTape() as tape:

        logits = discriminator_model(img)

        # ! Don't Know The Loss Function
        loss = loss_fn(y, logits)

        gradients = tape.gradient(loss, discriminator_model.trainable_weights)

    t1 = np.sqrt(np.sum(np.square(np.multiply(gradients, mask))))

    gp = np.mean(np.square(t1 - 1))

    return gp