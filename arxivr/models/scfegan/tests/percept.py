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

def complete_imgs(images, masks, generated):
    completed_images = np.zeros(np.shape(images))
    masks /= 255.
    cv2.imshow('generated', generated)
    cv2.waitKey(0)
    cv2.imshow('images', images)
    cv2.waitKey(0)
    patches = np.multiply(generated, masks)
    cv2.imshow('patches', patches)
    cv2.waitKey(0)
    print(np.unique(masks))
    reversed_mask = np.logical_not(masks)
    reversed_mask = reversed_mask.astype('int')
    print(np.unique(reversed_mask))

    completion = images * reversed_mask
    completed_images = np.add(patches, completion)
    completed_images = completed_images.astype(np.uint8)
    cv2.imshow('completed', completed_images)
    cv2.waitKey(0)

    return completed_images


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

ground_truth = cv2.imread('C:/Users/abtex/Desktop/me.jpg')
generated = cv2.imread('C:/Users/abtex/Desktop/me2.jpg')
ground_truth = cv2.resize(ground_truth, (256, 256))
generated = cv2.resize(generated, (256, 256))


mask = create_mask(ground_truth)
mask = np.expand_dims(mask, axis=-1)

cv2.imshow('mask', mask)
cv2.waitKey(0)

completed = complete_imgs(ground_truth, mask, generated)
ground_truth = np.expand_dims(ground_truth, axis=0)
generated = np.expand_dims(generated, axis=0)
completed = np.expand_dims(completed, axis=0)


feature_extractor = VGG16(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
gt_activs, nf = extract_features(feature_extractor, ground_truth)
gen_activs, _ = extract_features(feature_extractor, generated)
cmp_activs, _ = extract_features(feature_extractor, completed)

pl = 0

for i in range(len(gt_activs)):
    t1 = (np.sum(np.sum(np.subtract(gen_activs[i], gt_activs[i])))/nf[i])
    t2 = (np.sum(np.sum(np.subtract(cmp_activs[i], gt_activs[i])))/nf[i])

    pl += t1 + t2

print(pl)