import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from .utils import extract_features, complete_imgs


'''
    Calculates the per_pixel_loss between grount_truth image and generated image

    ground_truth : Ground truth image
    generated : Output of the generator for given ground truth image
    mask : binary mask for replacing the region
    alpha : hyperparameter deciding the weight of removed region

    returns :
            ppl : per_pixel loss
'''
def per_pixel_loss(ground_truth, generated, mask, alpha):
    nf = np.prod(np.shape(ground_truth[0]))

    t1 = np.sum(np.multiply(mask, np.subtract(generated, ground_truth)))/nf
    t2 = np.sum(np.multiply((1 - mask), np.subtract(generated, ground_truth)))/nf

    ppl = t1 + (alpha * t2)

    return ppl


'''
    Calculates perceptual_loss

    feature_extractor : VGG-16 instance pretrained on imagenet
    ground_truth : Ground truth image
    generated : Output of the generated for given ground truth image
    completed : Completed image, with the removed area as the output of generator
    and the non-removed area replaced with the grount_truth

    returns :
            pl : perceptual_loss
'''
def perceptual_loss(feature_extractor, ground_truth, generated, completed):
    gt_activs, nf = extract_features(feature_extractor, ground_truth)
    gen_activs, _ = extract_features(feature_extractor, generated)
    cmp_activs, _ = extract_features(feature_extractor, completed)

    pl = 0

    for i in range(len(gt_activs)):
        t1 = (np.sum(np.sum(np.subtract(gen_activs[i], gt_activs[i])))/nf[i])
        t2 = (np.sum(np.sum(np.subtract(cmp_activs[i], gt_activs[i])))/nf[i])

        pl += t1 + t2

    return pl


'''
    Calculates style loss

    feature_extractor : VGG-16 instance pretrained on imagenet
    gen : Output of generator for given ground truth image
    gt : Ground truth image

    returns :
            sl : style_loss
'''
def style_loss(feature_extractor, im, gt):
    gt_features, _ = extract_features(feature_extractor, gt)
    gen_features, _ = extract_features(feature_extractor, im)

    sl = 0

    for i in range(len(gen_features)):
        gt_feature = gt_features[i].reshape((-1, gt_features[i].shape[-1]))
        gen_feature = gen_features[i].reshape((-1, gen_features[i].shape[-1]))

        per_layer_features = gt_features[i].shape[-1] ** 2

        t1 = np.dot(gen_feature.T, gen_feature)
        t2 = np.dot(gt_feature.T, gt_feature)

        sl += np.sum((t1 - t2)/per_layer_features)

    return sl


'''
    Calculate variation loss between rows and columns and adds them together

    mask : binary_mask
    completed : Generated image with the non-removed region replaced with ground truth

    returns :
            tvl : total_variation_loss
'''
def total_variation_loss(masks, completed):
    completed = np.multiply(masks, completed)

    region = np.nonzero(completed)

    tvl_row = np.sum([np.sum(completed[i+1, j, :] - completed[i,j, :]) for i in region[0] for j in region[1]])
    tvl_row = tvl_row/np.size(completed)

    tvl_col = np.sum([np.sum(completed[i, j+1, :] - completed[i,j, :]) for i in region[0] for j in region[1]])
    tvl_col = tvl_col/np.size(completed)

    tvl = tvl_row + tvl_col

    return tvl


'''
    GSN Loss

    completed : Completed image with ground truth image and the non-removed replaced with ground truth

    returns :
            gsn : gsn loss
'''
def gsn_loss(y_pred):
    gsn = -1 * np.mean(y_pred)
    return gsn


'''
    Overall loss function for generator

    y_true : the ground truth image
    y_pred : generated image
    mask : binary mask
    feature_extractor : VGG 16 instance
    config : config object of the model

    returns :
            g_loss : Overall loss for the generator

'''
def generator_loss_function(y_true, y_pred, mask, feature_extractor, config):
    completed_image = complete_imgs(y_true, mask, y_pred)
    ppxl_loss = per_pixel_loss(y_true, y_pred, mask, config.HYPERPARAMETERS.ALPHA)
    perc_loss = perceptual_loss(feature_extractor, y_true, y_pred, completed_image)
    g_sn_loss = 0
    sg_loss = style_loss(feature_extractor, y_pred, y_true)
    sc_loss = style_loss(feature_extractor, completed_image, y_true)
    tv_loss = total_variation_loss(mask, completed_image)
    add_term = 0
    
    g_loss = ppxl_loss + (config.SCFEGAN_SIGMA * perc_loss)\
            + (config.SCFEGAN_BETA * g_sn_loss) + (config.SCFEGAN_GAMMA *\
            (sg_loss + sc_loss)) + (config.SCFEGAN_V * tv_loss) + add_term

    return g_loss
    
    
'''
    Gradient Penalty loss, derived from the paper : Improved Training of Wasserstein GANs

    y_true : Ground Truth Label For Discriminator
    y_pred : Predicted Output Of Discriminator
    averaged_samples : Random Weighted Average Of Completed And Ground Truth Images

    returns :
            gradient penalty loss
'''
# Credits : https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
def gp_loss(y_true, y_pred, averaged_samples):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr,
                            axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


'''
    Overall loss function for discriminator

    config : configuration object for the model
    images : ground truth images
    completed : output of the generator with the non-removed area replaced with the ground truth
    discriminator : instance object for the discriminator model

    returns :
            d_loss : overall loss of the discriminator
'''
def discriminator_loss_function(config, real_out, generated_out, gp_loss_val):
    d_loss = np.mean(1 - real_out) + np.mean(1 + generated_out) + config.HYPERPARAMETERS.SCFEGAN_THETA * gp_loss_val
    return d_loss