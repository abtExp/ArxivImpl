from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras.optimizers import Adam

from os import listdir, mkdir
from os.path import isdir
import sys
sys.path.insert(0, './utils/')


from utils import *
from enet_utils import *
from data_loaders import *
import numpy as np


# STREAM VARS
STREAM_OUT_PATH = './data/video/out.mp4'
STREAM_URL = 'http://10.49.205.144:8080/shot.jpg'
FPS = 20.0
IS_STREAMING = False
STREAM_FROM = 'file'
STREAM_SOURCE = './data/detector/train/video/video.mp4'
VIDEO_MAX_LENGTH_IN_SECONDS = 30


# GLOBALS
INP_SHAPE = (1024, 1024, 3)

# DETECTOR HYPER-PARAMETERS
DETECTOR_BATCH_SIZE = 16
DETECTOR_ORIGINAL_BATCH_SIZE = 8
DETECTOR_ITERATIONS = 3000
DETECTOR_IMPROVE_ITERATIONS = 6000
DETECTOR_EPOCHS = 100
DETECTOR_STEPS_PER_EPOCH = 60
DETECTOR_LR = 1e-4
DETECTOR_UNDISCOVERED_SIZE = 150000
DETECTOR_SYNTHETIC_IMAGES = 500
DETECTOR_MAX_ANCHORS = 5
BOXES_PER_GRID_CELL = 5
DETECTOR_MAX_DETECTIONS_PER_IMAGE = 10
DETECTOR_MAX_BOXES_PER_CELL = 5
DETECTOR_GRID_H = 32
DETECTOR_GRID_W = 32
DETECTOR_SYNTHETIC_MASK_MAX_POINTS = 5
DETECTOR_SYNTHETIC_MASK_PERT = 0.2
DETECTOR_LABEL_FILES_FORMAT = 'txt'

# TODO : Best anchors to be recalculated or use predefined from some other project
BEST_ANCHORS_1024 = [[4.85915839, 6.93810575], [0.98265896, 2.69475322], [10.23734818, 15.26189271], [1.28340875, 1.18959276], [2.55325957, 3.47480336]]
BEST_ANCHORS_416 = [[1.12,0.67], [2.19,1.90], [5.32,4.41], [5.44,1.71], [13.55,6.15]]
BEST_ANCHORS_512 = [[1.17,0.85], [3.23,3.82], [3.42,1.53], [8.34,4.39], [16.87,7.94]]
BEST_ANCHORS_256 = [[0.59,0.42], [1.61,1.91], [1.71,0.76], [4.17,2.20], [8.44,3.97]]

BEST_ANCHORS = np.reshape(BEST_ANCHORS_256,[-1])

DETECTOR_TRAIN_DATA_PATH = './data/detector/train/'
DETECTOR_TEST_DATA_PATH = './data/detector/test/'
DETECTOR_TEST_BATCH_SIZE = 25
DETECTOR_GET_BEST_ANCHOR_BOXES = get_best_anchor_boxes
DETECTOR_DATA_LOADER = load_data
DETECTOR_MASK_GENERATOR = generate_mask
DETECTOR_SYNTHETIC_DATA_GENERATOR = synthesize

# YOLO SPECIFIC HYPER-PARAMETERS
# YOLO_LOSS = yolo_loss
# YOLO_OUTPUT_FORMAT = yolo_outputs
YOLO_DATA_LOADER = load_valid_data
YOLO_CLASS_WEIGHTS = np.ones(2, dtype='float32')
YOLO_PRED_THRESHOLD = 0.5
YOLO_NMS_THRESHOLD = 0.45
YOLO_NO_OBJECT_SCALE = 1.0
YOLO_OBJECT_SCALE = 5.0
YOLO_COORD_SCALE = 1.0
YOLO_CLASS_SCALE = 1.0
YOLO_WARM_UP_BATCHES = 0
YOLO_CHECKPOINT_PATH = './checkpoints/checkpoints_yolo'
NUM_LANDMARK_POINTS = 98
FEATURE_POINTS = 6


# EXPERIMENTAL GENERATOR VARS
GEN_DATA_LOADER = GeneratorLoader
GEN_INP_SHAPE = (64, 64)
GEN_LATENT_DIM = 512
GEN_LR = 1e-3
GEN_BATCH_SIZE = 1
GEN_TEST_BATCH_SIZE = 4
GEN_EPOCHS = 6000
ENC_BATCH_SIZE = 32
GEN_LATENT_DIM = 512
ENC_CHECKPOINTS_PATH = './experiments/checkpoint_enc'
OR_CHECKPOINTS_PATH = './experiments/checkpoint_or'
LOGO_LOADER = logo_loader
RANDOMIZER = randomizer
CCNGAN_DATA_LOADER = ccngan_data_loader

# CGAN
GAN_DATA_LOADER = gen_loader
GEN_LOADER_TEST_BATCH_SIZE = 10

# RECAST SPECIFIC

# Whether to process target video as well
PROCESS_TARGET_VIDEO = False
# path for the stored meta data of target videos
META_PATH = './data/meta/'
# path for the result of recast
OUT_PATH = './data/out/'
# path for the saved muted audio from the target video
AUDIO_SAVE_PATH_TARGET = './data/audio/target/'
# path for saving the audio from the recorded video
AUDIO_SAVE_PATH_RECORDED = './data/audio/recorded/'
# path of the target video
VIDEO_PATH_TARGET = './data/video/target/'
# path of the recorded video
VIDEO_PATH_RECORDED = './data/video/recorded/'
# path to save the output audio
AUDIO_PATH = './data/audio/out/'
# path to save the output video
OUT_PATH = './data/out/'

AUDIO_OUTPUT_PATH = 'D:/recast/data/out/audio/'
VIDEO_OUTPUT_PATH = 'D:/recast/data/out/video/'

USER_NAMES = ['anubhav']
SOURCE_NAMES = ['messi']
PERSON_PATH = './faceit_live/data/persons/'
FACE_SHAPE = (512, 512)
VIDEO_PATH = './faceit_live/data/video/'
SOURCE_VIDEO = VIDEO_PATH + 'messi.mp4'

AUDIO_TRAIN_FILES_PATH_VOICE = './data/sound/mir/vocals/'
AUDIO_TRAIN_FILES_PATH_FULL_SAMPLES = './data/sound/mir/full_samples/'
AUDIO_TRAIN_FILES_PATH_FULL_FULL = './data/sound/mir/full_full/'

AUDIO_TRAIN_FILES_PATH = './data/sound/vad/train/'
AUDIO_VALID_FILES_PATH = './data/sound/vad/valid/'

CLOUD_FEATURE_PATH = 'D:/recast/data/export/train/features/'
CLOUD_LABEL_PATH = 'D:/recast/data/export/train/labels/'

INPUT_FILE_TYPE = 'wav'

TRAIN_EPOCHS = 100
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TEST_BATCH_SIZE = 8
# TEST_EPOCHS = 10

AUDIO_BATCH_SIZE = 16
VALID_AUDIO_BATCH_SIZE = 2
MAX_SEQUENCE_LENGTH = 4
MAX_TIME = 2.16 #2.154  #Pretty Odd Numbers, here, just to make the model possible
SEQUENCE_FEATURES = 21
# FRAME_RATE = 44000 # Next Option is 22k
FRAME_RATE = 16000
SAMPLE_STEP_SIZE = 2
N_FFT = 1023
FRAMES_PER_BUFFER = 1024
N_OVERLAP = FRAMES_PER_BUFFER // 2
N_BINS = FRAMES_PER_BUFFER // 2 + 1
# AUDIO_DATA_LOADER = AUDIO_DATA_LOADER
LABEL_TYPE = 'mask'
DEBUG = False
USE_CLOUD = False

PHASE_ITERATIONS = 10
# BEST_WEIGHT_PATH = './checkpoints/checkpoints_conv_12/weights.99-0.49.hdf5'
# BEST_WEIGHT_PATH = './checkpoints/checkpoints_conv_15/weights.74-0.57.hdf5'
# BEST_WEIGHT_PATH = './checkpoints/checkpoints_conv_18/weights.13-0.23.hdf5'
BEST_WEIGHT_PATH = './checkpoints/checkpoints_custom_0/weights.51-0.58.hdf5'

MODEL_IMAGE_PATH = './model_images/'
FACE_PRETRAINED_PATH = './checkpoints/checkpoints_face_vgg/wt.hdf5'
# FACE_DATA_LOADER = FACE_DATA_LOADER

ENET = {
	loader = enet_utils.data_loader
}

def get_callbacks(model='enet', max_time=MAX_TIME):
	if model == 'enet':
			formatter = enet_outputs
			plotter = plot_enet
	elif model == 'yolo':
			formatter = yolo_outputs
			plotter = plot

	all_checks = listdir('./checkpoints/')
	all_logs = listdir('./logs/')
	counter = 0
	max = -1

	for folder in all_checks:
			if 'checkpoints_{}'.format(model) in folder:
					if int(folder[folder.rindex('_')+1:]) > max:
							max = int(folder[folder.rindex('_')+1:])

	counter = max+1
	check_path = './checkpoints/checkpoints_{}_{}/'.format(model, counter)
	logs_path = './logs/logs_{}_{}/'.format(model, counter)

	if not isdir(check_path) and not isdir(logs_path):
			mkdir(check_path)
			mkdir(logs_path)



	# eval_cb = EvaluationCallback(model, INP_SHAPE, formatter, plotter, FLIKR_ONLY_LOGO_IDS)
	# eval_cb = EvalCallback(MAX_TIME, model)
	# evalcall = EvaluationCall(model, max_time, FRAME_RATE, N_FFT, LABEL_TYPE)
	checkpoint = ModelCheckpoint(check_path+'weights.{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=0, save_best_only=True)
	# earlystop = EarlyStopping(monitor='loss', min_delta=0, patience=3, verbose=0)
	tensorboard = TensorBoard(log_dir=logs_path, histogram_freq=0, batch_size=32, write_graph=True, write_grads=True, write_images=True)
	reducelr = ReduceLROnPlateau(monitor='loss', factor=0.02, patience=1, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)

	return [checkpoint, tensorboard, reducelr]