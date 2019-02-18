import numpy as np

import cv2

from os import listdir
from os.path import join
from PIL import Image, ImageOps

from lxml import etree

import matplotlib.pyplot as plt

from utils import kmeans, enet_utils

import gc

import keras

from sklearn.utils import shuffle

from matplotlib.path import Path
import matplotlib.patches as patches


# def mAP(y_true, y_pred):
#     # for all queries
#     APS = []
#     for i in range(len(queries)):
#         precisions, recalls = queries[i]
#         max_recall = 1
#         max_precision = 0

#         for i in range(len(precisions)-1, 0, -1):
#             max_precision = np.maximum(precisions[i-1], max_precision)
#             precisions[i-1] = max_precision

#         index = np.where(recalls[1:] != recalls[:-1])[0]

#         AP = np.sum((recalls[i+1] - recalls[i]) * precisions[i + 1])

#         APS.append(AP)

#     mAP = np.sum(APS)/len(APS)

#     return mAP


def iou(gt_box, box):
	intersection = 0
	union = 0

	area_box = box[1]*box[0]
	area_gt = gt_box[1]*gt_box[0]

	iou_ = intersection/union
	return iou_


def generate_mask(num_edges, pert, bbox, image):
    # pts = num_edges * 3 + 1
    # angles = np.linspace(0, 2*np.pi, pts)
    # codes = np.full(pts, Path.CURVE4)
    # codes[0] = Path.MOVETO

    # verts = np.stack((np.cos(angles)*0.5, np.sin(angles)*0.5)).T *(2*pert*np.random.random(pts)+1-pert)[:, None]
    # verts[-1, :] = verts[0, :]
    # path = Path(verts, codes)

    # fig = plt.figure(figsize=(bbox[2]/96, bbox[3]/96))
    # ax = fig.add_subplot(111)
    # ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    # ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)

    # patch = patches.PathPatch(path, facecolor='none', transform=ax.transData)

    if type(image) == 'str':
        image = Image.open(join('../data/detector/train/logos/',image))
        # image = Image.open(join('/logos',image))

    # image = image.resize((bbox[2], bbox[3]))
    # image = np.array(image)
    # ax.axis('off')
    # im = ax.imshow(image)
    # ax.add_patch(patch)
    # im.set_clip_path(patch)

    # fig.canvas.draw()

    # out = np.array(fig.canvas.renderer._renderer)
    # out = cv2.resize(out, (bbox[2], bbox[3]))
    image = image.convert("RGBA")
    datas = image.getdata()
    
    newData = []
    for item in datas:
        if item[0] == 255 and item[1] == 255 and item[2] == 255:
            newData.append(((255, 255, 255, 0)))
            
        else:
            newData.append(item)

    image.putdata(newData)

    image = image.resize((bbox[2], bbox[3]))
    image = np.array(image)
    out = np.transpose(image, (1, 0, 2))

    # plt.close()

    # return only RGB
    return out


def generate_combined_mask(masks, color_code_dict = {}):
    shape = (masks.shape[0], masks.shape[1])
    img = np.zeros(shape)   
    masks = masks.transpose()
    masks = np.transpose(masks, axes=(0,2,1))
    masks = masks.reshape((masks.shape[0], masks.shape[1]*masks.shape[2]))
    masks = masks/255

    img = np.reshape(img,[-1])

    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if img[j] == 0:
                img[j] = masks[i][j]
        

    img = np.reshape(img, shape)
    
    return img


def load_data(directory, batch_size=10, img_dim=(512, 512), grid_dim=(16,16), boxes_per_cell=5, num_classes=194, anchors=[], logo_class_dict={}, max_boxes=5, mode='mask', labels_files_format='txt', vars = {}):

    flikr_class_logos_dict = vars.FLIKR_NUM_LOGO_IDS

    img_files = [file for file in listdir(directory) if file.index('.') == len(file)-4 and file.endswith('png')]
    label_files = [file for file in listdir(directory) if file.endswith('.{}'.format(labels_files_format))]

    file_idxs = np.random.randint(0, len(img_files), batch_size)

    img_files = [img_files[i] for i in file_idxs]
    label_files = [j for i in img_files for j in listdir(directory) if j.endswith('.{}'.format(labels_files_format)) and i[:i.index('.')] in j]

    if mode == 'mask':
        all_mask_files = [file for file in listdir(directory) if file.endswith('png') and 'mask' in file]
        all_mask_files = [j for i in label_files for j in all_mask_files if i[:i.index('.')] in j]

    masks = []
    images = []
    labels = []

    # anchors = get_best_anchor_boxes(directory, max_boxes)

    for i in range(len(img_files)):

        if mode == 'mask':
            masks_for_this = [file for file in all_mask_files if img_files[i][:img_files[i].index('.')] in file]
            mask_label = np.zeros((img_dim[0], img_dim[1], num_classes))

        classes = np.zeros((num_classes))
        # label = np.zeros((grid_dim[0], grid_dim[1], boxes_per_cell, 5+num_classes))

        img = Image.open(join(directory, img_files[i])).resize((img_dim[0], img_dim[1]), Image.ANTIALIAS)
        # img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        img = np.array(img)
        images.append(np.array(img))

        grid_cell_size = img_dim[0]/grid_dim[0]

        data = []

        if labels_files_format == 'txt':
            with open(join(directory, label_files[i])) as f:
                data = f.read()
                data = data.split('\n')
                data = [i for i in data if len(i) > 0]

        elif labels_files_format == 'xml':
            xml = open(join(directory, label_files[i]))
            file = xml.read()
            tree = etree.fromstring(file)
            rect = {}

            for item in tree.iter('xmin', 'xmax', 'ymin', 'ymax', 'name'):
                if(item.tag in ['xmin','ymin','xmax','ymax']):
	                rect[item.tag] = int(item.text)

                if(item.tag == 'name'):
                    rect[item.tag] = int(logo_class_dict[item.text])

                if(len(rect) == 5):
                    data.append(rect)
                    rect = {}
            
            data = list(data.values())

        for j in range(len(data)):
            true_box_index = 0
            best_iou = anchors[true_box_index]
            box_data = data[j].split(' ')
            box_coords = box_data[0:4]
            box_class = int(box_data[4])

            # For converting brand_text and brand_symbol into the same class
            box_class_name = flikr_class_logos_dict[box_class]
            box_class_name = box_class_name if '_' not in box_class_name else box_class_name[:box_class_name.index('_')]

            box_class = logo_class_dict[box_class_name.lower()]

            if mode == 'mask':
                mask_i = np.array(Image.open(join(directory,masks_for_this[j])).resize((img_dim[0], img_dim[1]), Image.ANTIALIAS))/255
                for l in range(mask_i.shape[0]):
                    for k in range(mask_i.shape[1]):
                        if mask_label[l,k,box_class] == 0:
                            mask_label[l,k,box_class] = mask_i[l][k]
        
        combined_mask = generate_combined_mask(mask_label)
        last_mask = np.zeros((img_dim[0], img_dim[1]))
        for r in range(img_dim[0]):
            for c in range(img_dim[1]):
                if combined_mask[r,c] == 0:
                    last_mask[r,c] = 1

        mask_label[:,:,-1] = last_mask[:,:]


        #     classes[box_class] = 1

        #     center_x = ((int(box_coords[0]) + int(box_coords[2]))*0.5)/img_dim[0]
        #     center_y = ((int(box_coords[1]) + int(box_coords[3]))*0.5)/img_dim[1]
        #     box_h = (int(box_coords[2]) - int(box_coords[0]))/img_dim[0]
        #     box_w = (int(box_coords[3]) - int(box_coords[1]))/img_dim[1]

        #     grid_row = int(center_x/grid_cell_size)
        #     grid_col = int(center_y/grid_cell_size)

        #     box = [1, center_x, center_y, box_w, box_h]

        #     for i in range(len(anchors)):
        #         curr_iou = anchors[i] - (box_h * box_w)
        #         if curr_iou < best_iou:
        #             best_iou = curr_iou
        #             true_box_index = i

        #     label[grid_row, grid_col, true_box_index, :] = [*box, *classes]
        # labels.append(label)
        if mode == 'mask':
            mask_label = np.reshape(mask_label, (img_dim[0]*img_dim[1], num_classes))
            masks.append(mask_label)
            # generate_combined_mask(mask_label)

    # Leave it here to deal with grayscale
    images = np.reshape(images, (len(images), img_dim[0], img_dim[1], img_dim[2]))

    return np.array(images), np.array(labels), np.array(masks)

def bbox_from_mask(masks):
	boxes = []

	for i in range(np.shape(masks)[-1]):
		mask = masks[:,:,i]

		horizontal_idxs = np.where(np.any(mask, axis=0))[0]
		vertical_idxs = np.where(np.any(mask, axis=1))[0]

		if horizontal_idxs.shape[0]:
			x1, x2 = horizontal_idxs[[0,-1]]
			y1, y2 = vertical_idxs[[0, -1]]
			x2 = x2 + 1
			y2 = y2 + 1

		else:
			x1, x2, y1, y2 = 0,0,0,0

		boxes.append([x1, y1, x2, y2])

	return np.array(boxes)

def get_best_anchor_boxes(train_path, num_clusters, mode='txt'):


    labeled_images = [example for example in listdir(train_path) if example.endswith('.{}'.format(mode))]
    rectangles = []


    if mode == 'xml':
        for file in labeled_images:
            xml = open(join(train_path, file))
            file = xml.read()
            tree = etree.fromstring(file)
            rect = {}
            for item in tree.iter('xmin', 'xmax', 'ymin', 'ymax', 'name'):
                if item.tag in ['xmin', 'xmax', 'ymin', 'ymax']:
                    rect[item.tag] = int(item.text)

                if(len(rect) == 5):
                    rectangles.append(rect)
                    rect = {}

        rectangles = [[(rect['xmax']-rect['xmin']), (rect['ymax']-rect['ymin'])] for rect in rectangles]

    elif mode == 'txt':
        for i in range(len(labeled_images)):
            with open(join(train_path, labeled_images[i])) as f:
                data = f.read()
                data = data.split('\n')
                for i in data:
                    if len(i) > 0:
                        box_data = i.split(' ')
                        box_coords = box_data[0:4]
                        box_h = (int(box_coords[2]) - int(box_coords[0]))
                        box_w = (int(box_coords[3]) - int(box_coords[1]))
                        rectangles.append([box_w, box_h])
                del data
                gc.collect()
    else:
        print('Invalid Mode.')
    
    best_anchors = kmeans.kmeans(rectangles, num_clusters)

    return best_anchors


# generate new training data for synthetic learning with random patches of logos placed inside random bboxes
def synthesize(train_path, inp_shape=(512,512,3), grid_shape=(16,16), num_classes=194, num_boxes=5, anchors=[], max_pts=5, pert=0.4, logo_class_dict={}, num_images=0, background_dir='./flikr/non_logo', backgrounds=None, logos=None, background_idxs=None, logo_idxs=None):
    images = []
    labels = []
    mask_labels = []

    background_dir = join(train_path, background_dir)
    logo_dir = join(train_path, './logos')

    # logo_dir = '/logos'

    all_logos = [i for i in listdir(logo_dir) if i.endswith('jpg') and i[:i.index('.')] in logo_class_dict.keys()]

    # logo_class from substring
    logo_classes = [int(logo_class_dict[i[:i.index('.')]]) for i in all_logos]

    if backgrounds == None:
        if background_idxs == None:
            # getting random backgrounds for synthesizing images
            background_idxs = np.random.rand((num_images))*len(listdir(background_dir))
            background_idxs = [int(a) for a in background_idxs]

        backgrounds = [Image.open(join(background_dir, i)).resize(tuple(inp_shape[0:2])) for i in [listdir(background_dir)[int(j)] for j in background_idxs]]

    if logos == None:
        if logo_idxs == None:
            # getting random logos for synthesizing images
            logo_idxs = np.random.rand((num_images))*len(all_logos)
            logo_idxs = [int(a) for a in logo_idxs]

        logos = [Image.open(join(logo_dir,j)) for j in [all_logos[k] for k in logo_idxs]]

    grid_cell_size = inp_shape[0]/grid_shape[0]

    for i, image in enumerate(backgrounds):
        # generating random bounding boxes
        # image = np.array(image)
        shape = np.shape(image)
        bbox = [
            np.random.randint(50,shape[0]-50), 
            np.random.randint(50,shape[0]-50),
            np.random.randint(20, 100),
            np.random.randint(20, 100)
            ]
        

        true_box_index = 0
        # best_iou = anchors[true_box_index]

        # for anc in range(len(anchors)):
        #     curr_iou = anchors[anc] - (bbox[2] * bbox[3])
        #     if curr_iou < best_iou:
        #         best_iou = curr_iou
        #         true_box_index = anc
        
        mask = np.zeros((inp_shape[0], inp_shape[1], num_classes))

        logo = logos[i].resize((bbox[2], bbox[3]), Image.ANTIALIAS)
        
        logo = generate_mask(max_pts, pert, bbox, logo)
        # logo_activ = cv2.cvtColor(logo, cv2.COLOR_RGB2GRAY)
        # logo_activ = logo_activ/255
        # print('THE LOGO')
        # plt.imshow(logo_activ, cmap='gray')
        # plt.show()
        w=0
        h=0

        image = image.convert('RGBA')
        image = np.array(image)

        for y in range(bbox[0] - int(bbox[2]*0.5), (bbox[0] + int(bbox[2]*0.5))-1, 1):
            w = 0
            for x in range(bbox[1] - int(bbox[3]*0.5), (bbox[1] + int(bbox[3]*0.5))-1, 1):
                if logo[h,w,3] != 0:
                    image[x,y,:] = logo[h,w,:]
                    mask[x,y,logo_classes[i]] = 255
                w = w+1
            h = h+1
        
        last_mask = np.zeros((inp_shape[0], inp_shape[1]))
        for r in range(inp_shape[0]):
            for c in range(inp_shape[1]):
                if mask[r,c,logo_classes[i]] == 0:
                    last_mask[r,c] = 255
        
        mask[:,:,-1] = last_mask

        # finding relative position of the bbox wrt the grid
        # grid_row_num = bbox[0]/grid_cell_size
        # grid_col_num = bbox[1]/grid_cell_size

        # grid_row_num = np.floor(grid_row_num) if grid_row_num - \
        #                         np.floor(grid_row_num) > 0 else grid_row_num
        # grid_col_num = np.floor(grid_col_num) if grid_col_num - \
        #                         np.floor(grid_col_num) > 0 else grid_col_num

        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2RGB)

        image = np.array(image)
        # image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        # image = np.array(image)
        images.append(image)

        img_class = np.zeros((num_classes))

        img_class[logo_classes[i]] = 1

        # label = [1, *bbox, *img_class]

        # boxes = np.zeros((grid_shape[0], grid_shape[1],
        #                         num_boxes, 1+4+num_classes))

        # boxes[int(grid_row_num)-1, int(grid_col_num)-1, true_box_index] = label


        # labels.append(boxes)
        mask = np.reshape(mask, (inp_shape[0]*inp_shape[1], num_classes))
        mask_labels.append(mask)
    images = np.reshape(images, (len(images), inp_shape[0], inp_shape[1], inp_shape[2]))
    return np.array(images), np.array(labels), np.array(mask_labels)


def eval(model, vars):
    imgs = listdir(vars.EVAL_DICT)
    idxs = np.random.randint(0, len(imgs), vars.EVAL_BATCH_SIZE)
    eval_x = np.array([np.array(Image.open(join(vars.EVAL_DICT, imgs[i])).resize((vars.INP_SHAPE[0], vars.INP_SHAPE[1]))) for i in idxs])

    masks, preds, scores, scores_per_class = model.predict(eval_x)

    return masks, preds, scores, scores_per_class

class BootStrapGenerator(keras.utils.Sequence):
    def __init__(self, original_batch_size, synthetic_batch_size, vars, mode='train'):
        self.inp_shape = vars.INP_SHAPE
        self.vars = vars
        self.labels_files_format = vars.DETECTOR_LABEL_FILES_FORMAT
        self.original_bach_size = original_batch_size
        self.synthetic_batch_size = synthetic_batch_size
        self.batch_size = original_batch_size + synthetic_batch_size
        self.mode = mode
    
    def __getitem__(self, index):
        
        x, mask = self.__data_generation(None)

        index = np.random.randint(0, len(x))
        
        x = x[index]
        # y = y[index]
        mask = mask[index]
        
        x = np.reshape(x, (1, np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]))
        # y = np.reshape(y, (1, np.shape(y)[0], np.shape(y)[1], np.shape(y)[2], np.shape(y)[3]))
        mask = np.reshape(mask, (1, np.shape(mask)[0], np.shape(mask)[1]))

        return x, mask

    def __len__(self):
        return self.vars.DETECTOR_BATCH_SIZE

    def __data_generation(self, a):
        # if self.mode == 'train':
        directory = '{}/flikr/{}'.format(self.vars.DETECTOR_TRAIN_DATA_PATH, self.mode)
        # else:
        #     directory = '/valid'

        original_x, original_y, original_mask = load_data(
                                                    directory,
                                                    batch_size=self.original_bach_size,
                                                    img_dim=self.vars.INP_SHAPE, 
                                                    grid_dim =(self.vars.DETECTOR_GRID_W, self.vars.DETECTOR_GRID_H),
                                                    boxes_per_cell = self.vars.DETECTOR_MAX_BOXES_PER_CELL, 
                                                    num_classes=self.vars.LOGO_NUM_CLASSES, 
                                                    logo_class_dict=self.vars.FLIKR_ONLY_LOGO_CLASS_DICT, 
                                                    max_boxes=self.vars.DETECTOR_MAX_ANCHORS, 
                                                    anchors=self.vars.BEST_ANCHORS,
                                                    mode='mask',
                                                    vars=self.vars
                                                )
        if self.mode == 'train':
            synthetic_x, synthetic_y, synthetic_mask = synthesize(
                                                        self.vars.DETECTOR_TRAIN_DATA_PATH,
                                                        self.vars.INP_SHAPE,
                                                        (self.vars.DETECTOR_GRID_W, self.vars.DETECTOR_GRID_H),
                                                        self.vars.LOGO_NUM_CLASSES,
                                                        self.vars.DETECTOR_MAX_ANCHORS,
                                                        self.vars.BEST_ANCHORS,
                                                        max_pts=self.vars.DETECTOR_SYNTHETIC_MASK_MAX_POINTS,
                                                        pert=self.vars.DETECTOR_SYNTHETIC_MASK_PERT,
                                                        logo_class_dict=self.vars.FLIKR_ONLY_LOGO_CLASS_DICT,
                                                        num_images = self.synthetic_batch_size
                                                )
            train_x = np.array([*original_x, *synthetic_x])
            # train_y = np.array([*original_y, *synthetic_y])
            train_mask = np.array([*original_mask, *synthetic_mask])
        
        else:
            train_x = np.array(original_x)
            train_mask = np.array(original_mask)

        return train_x, train_mask


def yolo_target_from_enet(masks, vars, anchors):
    labels = []
    for i in range(len(masks)):
        label = np.zeros((vars.DETECTOR_GRID_H, vars.DETECTOR_GRID_W, vars.MAX_BOXES_PER_CELL, 1+4+vars.LOGO_NUM_CLASSES))
        classes = np.zeros((vars.LOGO_NUM_CLASSES))
        for mask in masks[i]:
            mask = np.reshape(mask, (vars.INP_SHAPE[0], vars.INP_SHAPE[1], vars.LOGO_NUM_CLASSES))
            for j,m in enumerate(mask):
                bbox = bbox_from_mask(m)
                
                classes[j] = 1

                center_x = ((int(bbox[0]) + int(bbox[2]))*0.5)/vars.INP_SHAPE[0]
                center_y = ((int(bbox[1]) + int(bbox[3]))*0.5)/vars.INP_SHAPE[1]
                box_h = (int(bbox[2]) - int(bbox[0]))/vars.INP_SHAPE[0]
                box_w = (int(bbox[3]) - int(bbox[1]))/vars.INP_SHAPE[1]

                grid_row = int(center_x/vars.DETECTOR_GRID_H)
                grid_col = int(center_y/vars.DETECTOR_GRID_W)

                box = [1, center_x, center_y, box_w, box_h]

                for k in range(len(anchors)):
                    curr_iou = anchors[k] - (box_h * box_w)
                    if curr_iou < best_iou:
                        best_iou = curr_iou
                        true_box_index = k

                label[grid_row, grid_col, true_box_index, :] = [*box, *classes]
            labels.append(label)

            
    return labels

def apply_random_alpha(image):
    image = np.array(image)
    np.random.seed(100)
    alpha = np.random.randint(170, 255)
    image[:,:,-1] = alpha
    return Image.fromarray(image)

def apply_random_crop(image):
    shape = np.shape(image)
    random_patch_dims = (
            np.random.randint(0, 10), #x0
            np.random.randint(0, 10), #y0
            np.random.randint(shape[0]-20, shape[0]), #x1
            np.random.randint(shape[1]-20, shape[1])  #y1
        )

    image = image.crop(random_patch_dims)

    image = image.resize((shape[0], shape[1]))

    return image

def randomizer(batch_size):
    path = './data/detector/train/flikr/train'
    train_x = []
    train_y = []

    rotations = np.arange(0,360,15)

    image_files = [file[:file.index('.')] for file in listdir(path) if file.index('.') == len(file)-4 and file.endswith('png')]
    indices = np.random.randint(0, len(image_files), batch_size)
    image_files = [image_files[i] for i in indices]

    for file in image_files:
        image = Image.open('{}/{}.png'.format(path,file))
        with open('{}/{}.gt_data.txt'.format(path, file)) as f:
            bboxes = f.read()
            bboxes = bboxes.split('\n')
            bboxes = [i for i in bboxes if len(i) > 0]
            bboxes = [i.split(' ') for i in bboxes]

        for bbox in bboxes:
            box_coords = [int(index) for index in bbox[0:4]]
            img = image.crop(box_coords)
            img = img.convert('LA')
            img = apply_random_alpha(img)
            angles = np.random.choice(rotations, 3, replace=False)
            indices = [i for j in angles for i in range(len(rotations)) if j == rotations[i]]

            for i, angle in enumerate(angles):
                img = img.rotate(angle)
                img = img.resize((64,64), Image.ANTIALIAS)
                label = np.zeros(24)
                label[indices[i]] = 1
                train_x.append(np.array(img))
                train_y.append(label)
    return shuffle(train_x, train_y)

def logo_loader(logo_dict, batch_size):
    train_x = []
    train_y = []
    path = './data/detector/train/logos'
    images = listdir(path)

    idxs = np.random.randint(0, len(images), batch_size)
    images = [images[i] for i in idxs]

    for image in images:
        img = Image.open(join(path, image))
        img = generate_mask(None, None, [0,0,64,64], img)
        plt.imshow(img)
        plt.show()
        train_x.append(np.array(img))
        label = np.zeros(194)
        label[logo_dict[image[:image.index('.')].lower()]] = 1
        train_y.append(label)
    
    return shuffle(np.array(train_x), np.array(train_y))
class GeneratorLoader(keras.utils.Sequence):
    def __init__(self, vars, mode='enc'):
        self.mode = mode
        self.vars = vars
        self.batch_size = self.vars.ENC_BATCH_SIZE

    def __getitem__(self, index):
        x, y = self.__data_generation(None)

        index = np.random.randint(0, self.batch_size)
        
        x = x[index]
        y = y[index]
        
        x = np.reshape(x, (1, np.shape(x)[0], np.shape(x)[1], np.shape(x)[2]))
        y = np.reshape(y, (1, np.shape(y)[0]))

        return x, y

    def __len__(self):
        return len(listdir('./data/detector/train/logos'))

    def __data_generation(self, a):
        if self.mode == 'enc':
            return logo_loader(self.vars.LOGO_CLASS_IDS, self.batch_size)
        
        elif self.mode == 'or':
            return randomizer(self.batch_size)


class EvalCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        model = self.model

        eval_dir = './data/detector/test/logo+/'
        images = listdir(eval_dir)
        image_idx = np.random.randint(0, len(images))
        image = Image.open(join(eval_dir, images[image_idx])).resize((256, 256))
        image = np.array(image)
        cv2.imwrite('./eval_imgs/img.{}.png'.format(epoch), image)
        image = np.reshape(image,(1,256,256,3))
        color_dict = []

        masks, preds, scores, scores_per_class = enet_utils.enet_outputs(model.predict(image), (256, 256, 3), 33)
        image = enet_utils.plot(preds, scores_per_class, color_dict)
        image = np.reshape(image, (256, 256, 1))
        cv2.imwrite('./eval_imgs/img.{}.mask.png'.format(epoch), image)

