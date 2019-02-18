from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LambdaCallback
from keras.optimizers import Adadelta, Adam

from utils import yolo_utils, enet_utils, utils, dataloader, ieee_utils	
import numpy as np

# callbacks
checkpoint = ModelCheckpoint('./checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0, save_best_only=True)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0)
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
reducelr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
enc_check = ModelCheckpoint('./experiments/checkpoint_enc/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
or_check = ModelCheckpoint('./experiments/checkpoint_or/weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
earlystop_loss = EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=0)
eval_enet = utils.EvalCallback()

CALLBACKS = [checkpoint, tensorboard, reducelr, eval_enet]

# GLOBALS
INP_SHAPE = (256, 256, 3)
LOGO_ID = 12
LOGO_NUM_CLASSES = 33
FRAME_NAME = 'feed'
LOGO_REPLACE_DICT = {}
LOGO_CLASS_IDS = {'cisco': 0, 'pepsi': 1, 'kelloggs': 2, 'youtube': 3, 'stellaartois': 4, 'recycling': 5, 'sega': 6, 'republican': 7, 'walmart': 8, 'ebay': 9, 'becks': 10, 'lg': 11, 'toyota': 12, 'cvs': 13, 'guinness': 14, 'bmw': 15, 'singha': 16, 'underarmour': 17, 'timberland': 18, 'ikea': 19, 'dunkindonuts': 20, 'alfaromeo': 21, 'northface': 22, 'nasa': 23, 'suzuki': 24, 'carters': 25, 'calvinklein': 26, 'schwinn': 27, 'danone': 28, 'supreme': 29, 'bbc': 30, 'redbull': 31, 'chiquita': 32, 'esso': 33, 'converse': 34, 'soundrop': 35, 'soundcloud': 36, 'evernote': 37, 'porsche': 38, 'samsung': 39, 'windows': 40, 'poloralphlauren': 41, 'visa': 42, 'chevron': 43, 'jackinthebox': 44, '3m': 45, 'basf': 46, 'santander': 47, 'vaio': 48, 'tissot': 49, 'cartier': 50, 'texaco': 51, 'bacardi': 52, 'kraft': 53, 'spiderman': 54, 'coach': 55, 'marlboro': 56, 'chanel': 57, 'costco': 58, 'motorola': 59, 'zara': 60, 'lego': 61, 'mastercard': 62, 'fosters': 63, 'mobil': 64,'gap': 65, 'amazon': 66, 'microsoft': 67, 'hsbc': 68, 'kfc': 69, 'bayer': 70, 'disney': 71, 'louisvuitton': 72, 'barclays': 73, 'fedex': 74, 'hermes': 75, 'internetexplorer': 76, 'jagermeister': 77, 'citi': 78, 'tacobell': 79, 'londonunderground': 80, 'volkswagen': 81, 'wordpress': 82, 'lexus': 83, 'corona': 84, 'erdinger': 85, 'playstation': 86, 'android': 87, 'hyundai': 88, 'gucci': 89, 'pampers': 90, 'jcrew': 91, 'nescafe': 92, 'paulaner': 93, 'ford': 94, 'loreal': 95,'adidas': 96, 'skechers': 97, 'drpepper': 98, 'michelin': 99, 'xbox': 100, 'bottegaveneta': 101, 'mitsubishi': 102, 'bbva': 103, 'uniqlo': 104, 'hershey': 105, 'oracle': 106, 'caterpillar': 107, 'blizzardentertainment': 108, 'heineken': 109, 'chimay': 110, 'reebok': 111, 'honda': 112, 'nvidia': 113, 'superman': 114,'chevrolet': 115, 'netflix': 116, 'dhl': 117, 'prada': 118, 'shell': 119, 'lacoste': 120, 'espn': 121, 'thomsonreuters': 122, 'olympics': 123, 'sap': 124, 'starbucks': 125, 'rbc': 126, 'warnerbros': 127, 'teslamotors': 128, 'ferrari': 129, 'goodyear': 130, 'nintendo': 131, 'subaru': 132, 'panasonic': 133, 'asus': 134, 'generalelectric': 135, 'obey': 136, 'bershka': 137, 'nbc': 138, 'volvo': 139, 'maserati': 140, 'wellsfargo': 141, 'puma': 142, 'chickfila': 143, 'carlsberg': 144, 'cocacola': 145, 'mk': 146, 'philips': 147, 'hh': 148, 'hp': 149, 'mcdonald': 150, 'comedycentral': 151, 'nestle': 152, 'intel': 153, 'ups': 154, 'kodak': 155, 'tsingtao': 156, 'lamborghini': 157, 'wii': 158, 'firefox': 159, 'armani': 160, 'milka': 161, 'gildan': 162, 'bulgari': 163, 'unitednations': 164, 'rittersport': 165, 'nike': 166, 'millerhighlife': 167, 'nissan': 168, 'allianz': 169, 'barbie': 170, 'renault': 171, 'mtv': 172, 'fritolay': 173, 'homedepot': 174, 'subway': 175, 'boeing': 176, 'facebook': 177, 'hanes': 178, 'siemens': 179, 'apple': 180, 'bankofamerica': 181, 'aldi': 182, 'canon': 183, 'google': 184, 'levis': 185, 'colgate': 186, 'yamaha': 187, 'rolex': 188, 'johnnywalker': 189, 'luxottica': 190, 'batman': 191, 'gillette': 192, 'tommyhilfiger': 193, '__empty__':194}
FLIKR_LOGO_CLASS_IDS = {'HP': 0, 'adidas_symbol': 1, 'adidas_text': 2, 'aldi': 3, 'apple': 4, 'becks_symbol': 5, 'becks_text': 6, 'bmw': 7, 'carlsberg_symbol': 8, 'carlsberg_text': 9, 'chimay_symbol': 10, 'chimay_text': 11, 'cocacola': 12, 'corona_symbol': 13, 'corona_text': 14, 'dhl': 15, 'erdinger_symbol': 16, 'erdinger_text': 17, 'esso_symbol': 18, 'esso_text': 19, 'fedex': 20, 'ferrari': 21, 'ford': 22, 'fosters_symbol': 23, 'fosters_text': 24, 'google': 25, 'guinness_symbol': 26, 'guinness_text': 27, 'heineken': 28, 'milka': 29, 'nvidia_symbol': 30, 'nvidia_text': 31, 'paulaner_symbol': 32, 'paulaner_text': 33, 'pepsi_symbol': 34, 'pepsi_text': 35, 'rittersport': 36, 'shell': 37, 'singha_symbol': 38, 'singha_text': 39, 'starbucks': 40, 'stellaartois_symbol': 41, 'stellaartois_text': 42, 'texaco': 43, 'tsingtao_symbol': 44, 'tsingtao_text': 45, 'ups': 46}
FLIKR_NUM_LOGO_IDS = {0: 'HP', 1: 'adidas_symbol', 2: 'adidas_text', 3: 'aldi', 4: 'apple', 5: 'becks_symbol', 6: 'becks_text', 7: 'bmw', 8: 'carlsberg_symbol', 9: 'carlsberg_text', 10: 'chimay_symbol', 11: 'chimay_text', 12: 'cocacola', 13: 'corona_symbol', 14: 'corona_text', 15: 'dhl', 16: 'erdinger_symbol', 17: 'erdinger_text', 18: 'esso_symbol', 19: 'esso_text', 20: 'fedex', 21: 'ferrari', 22: 'ford', 23: 'fosters_symbol', 24: 'fosters_text', 25: 'google', 26: 'guinness_symbol', 27: 'guinness_text', 28: 'heineken', 29: 'milka', 30: 'nvidia_symbol', 31: 'nvidia_text', 32: 'paulaner_symbol', 33: 'paulaner_text', 34: 'pepsi_symbol', 35: 'pepsi_text', 36: 'rittersport', 37: 'shell', 38: 'singha_symbol', 39: 'singha_text', 40: 'starbucks', 41: 'stellaartois_symbol', 42: 'stellaartois_text', 43: 'texaco', 44: 'tsingtao_symbol', 45: 'tsingtao_text', 46: 'ups'}
FLIKR_ONLY_LOGO_CLASS_DICT = {'pepsi': 0, 'stellaartois': 1, 'becks': 2, 'guinness': 3, 'bmw': 4, 'singha': 5, 'esso': 6, 'texaco': 7, 'fosters': 8, 'fedex': 9, 'corona': 10, 'erdinger': 11, 'paulaner': 12, 'ford': 13, 'adidas': 14, 'heineken': 15, 'chimay': 16, 'nvidia': 17, 'dhl': 18, 'shell': 19, 'starbucks': 20, 'ferrari': 21, 'carlsberg': 22, 'cocacola': 23, 'hp': 24, 'ups': 25, 'tsingtao': 26, 'milka': 27, 'rittersport': 28, 'apple': 29, 'aldi': 30, 'google': 31, '__empty__': 32}

# RANDOMLY GENERATED COLOR CODES
LOGO_CLASS_COLOR_DICT = [
       [229, 150, 139],
       [127, 232, 125],
       [112, 227,  68],
       [238, 200, 100],
       [ 18,  43, 240],
       [  1,  96,  63],
       [ 84,  65, 158],
       [ 63,  91,  92],
       [249,  83,  32],
       [ 75, 217, 226],
       [ 53,  76, 163],
       [245, 188, 192],
       [137, 214,  59],
       [ 71, 126, 233],
       [145,   9,  80],
       [172, 133,   1],
       [213, 136,  87],
       [ 59,  89,  14],
       [ 72,  67, 176],
       [ 15,  62, 182],
       [248, 185, 141],
       [165,   6, 101],
       [215, 182,  35],
       [194, 184, 137],
       [ 49,  16,   6],
       [ 94, 215,  27],
       [ 97, 162,  91],
       [162, 109,  42],
       [ 39, 109, 169],
       [ 38, 156, 143],
       [188,  89, 169],
       [109, 101,  80],
       [ 0,  0, 0]
	]


# DETECTOR HYPER-PARAMETERS
DETECTOR_BATCH_SIZE = 16
DETECTOR_ORIGINAL_BATCH_SIZE = 8
DETECTOR_CALLBACKS = [checkpoint, earlystop, tensorboard, reducelr]
DETECTOR_ITERATIONS = 3000
DETECTOR_IMPROVE_ITERATIONS = 6000
DETECTOR_EPOCHS = 100
DETECTOR_STEPS_PER_EPOCH = 60
DETECTOR_LR = 1e-4
DETECTOR_UNDISCOVERED_SIZE = 150000
DETECTOR_SYNTHETIC_IMAGES = 500
DETECTOR_MAX_ANCHORS = 5
DETECTOR_MAX_DETECTIONS_PER_IMAGE = 10
DETECTOR_MAX_BOXES_PER_CELL = 5
DETECTOR_GRID_H = 16
DETECTOR_GRID_W = 16
DETECTOR_SYNTHETIC_MASK_MAX_POINTS = 5
DETECTOR_SYNTHETIC_MASK_PERT = 0.2
DETECTOR_LABEL_FILES_FORMAT = 'txt'
DETECTOR_BOOTSTRAP_GENERATOR = utils.BootStrapGenerator

# for cloud training change it to the data source bucket id and stream data from it
# DETECTOR_TRAIN_DATA_PATH = '/floyd/input/detector/detector/train'
DETECTOR_TRAIN_DATA_PATH = './data/detector/train'
DETECTOR_TEST_DATA_PATH = './data/detector/test/logo+/'
DETECTOR_TEST_BATCH_SIZE = 8
DETECTOR_GET_BEST_ANCHOR_BOXES = utils.get_best_anchor_boxes
DETECTOR_DATA_LOADER = utils.load_data
DETECTOR_MASK_GENERATOR = utils.generate_mask
DETECTOR_SYNTHETIC_DATA_GENERATOR = utils.synthesize

# use mAP instead
DETECTOR_METRICS = ['val_acc']
DETECTOR_OPTIMIZER = Adam()

# YOLO SPECIFIC HYPER-PARAMETERS
YOLO_LOSS = yolo_utils.yolo_loss
YOLO_OUTPUT_FORMAT = yolo_utils.yolo_outputs
YOLO_DATA_LOADER = yolo_utils.load_data
YOLO_CLASS_WEIGHTS = np.ones(LOGO_NUM_CLASSES, dtype='float32')
YOLO_PRED_THRESHOLD = 0.5
YOLO_NMS_THRESHOLD = 0.45
YOLO_NO_OBJECT_SCALE = 1.0
YOLO_OBJECT_SCALE = 5.0
YOLO_COORD_SCALE = 1.0
YOLO_CLASS_SCALE = 1.0
YOLO_WARM_UP_BATCHES = 0

# ENET SPECIFIC HYPER-PARAMETERS
ENET_OUTPUT_FORMAT = enet_utils.enet_outputs
ENET_LOSS = enet_utils.enet_loss

# CYCLE-GAN SPECIFIC VARS
CYCLE_GAN_DATA_LOADER = dataloader.DataLoader

# EXPERIMENTAL GENERATOR VARS
GEN_DATA_LOADER = utils.GeneratorLoader
GEN_INP_SHAPE = (64, 64, 3)
GEN_LR = 1e-3
GEN_BATCH_SIZE = 10
GEN_TEST_BATCH_SIZE = 4
GEN_EPOCHS = 6000
ENC_BATCH_SIZE = 32
ENC_CHECKPOINTS_PATH = './experiments/checkpoint_enc'
OR_CHECKPOINTS_PATH = './experiments/checkpoint_or'
LOGO_LOADER = utils.logo_loader
RANDOMIZER = utils.randomizer


# EXPERIMENT, IEEE
MUTUAL_ENHANCEMENT = ieee_utils.mutual_enhancement
SIBLING_SELECTION = ieee_utils.sibling_selection
CONSISTENCY = ieee_utils.calc_color_consistency
ALIGNMENT = ieee_utils.alignment
INHOMOGENEOUS = ieee_utils.inhomogeneous
FINAL_SCORE = ieee_utils.return_final_score