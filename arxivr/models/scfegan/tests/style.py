from tensorflow.keras.applications.vgg16 import VGG16
import tensorflow.keras.backend as K
import numpy as np

import cv2
import face_recognition
import math

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

def extract_features(feature_extractor, x):
    nf = []
    outputs = []
    functors = []

    for layer in feature_extractor.layers:
        if layer.name in ["block1_pool", "block2_pool", "block3_pool"]:
            nf.append(np.prod(layer.output_shape[1:]))
            functors.append(layer.output)

    appl = K.function([feature_extractor.input, K.learning_phase()], functors)
    outputs = appl([x, 1.])
    outputs = [out[0] for out in outputs]

    return outputs, nf

gt = cv2.imread('C:/Users/abtex/Desktop/me.jpg')
gen = cv2.imread('C:/Users/abtex/Desktop/me2.jpg')
gt = cv2.resize(gt, (256, 256))
gen = cv2.resize(gen, (256, 256))

gt = np.expand_dims(gt, axis=0)
gen = np.expand_dims(gen, axis=0)

feature_extractor = VGG16(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
gt_features, _ = extract_features(feature_extractor, gt)
gen_features, _ = extract_features(feature_extractor, gen)


sl = 0

for i in range(len(gen_features)):
    gt_feature = gt_features[i].reshape((-1, gt_features[i].shape[-1]))
    gen_feature = gen_features[i].reshape((-1, gen_features[i].shape[-1]))

    per_layer_features = gt_features[i].shape[-1] ** 2

    t1 = np.dot(gen_feature.T, gen_feature)
    t2 = np.dot(gt_feature.T, gt_feature)

    sl += np.sum((t1 - t2)/per_layer_features)

print(sl)