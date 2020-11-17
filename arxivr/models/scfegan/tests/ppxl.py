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

ground_truth = cv2.imread('C:/Users/abtex/Desktop/me.jpg')
generated = cv2.imread('C:/Users/abtex/Desktop/me2.jpg')
ground_truth = cv2.resize(ground_truth, (256, 256))
generated = cv2.resize(generated, (256, 256))
mask = create_mask(ground_truth)
mask = np.expand_dims(mask, axis=-1)


cv2.imshow('mask', mask)
cv2.waitKey(0)

nf = np.prod(np.shape(ground_truth[0]))

t1 = np.sum(np.multiply(mask, np.subtract(generated, ground_truth)))/nf
t2 = np.sum(np.multiply((1 - mask), np.subtract(generated, ground_truth)))/nf

ppl = t1 + (1.5 * t2)

print(ppl)

