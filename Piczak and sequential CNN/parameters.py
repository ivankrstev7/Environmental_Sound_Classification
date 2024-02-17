import tensorflow as tf

###############
### GENERAL ###
###############

WHICH_DATASET = 'esc50' # 'esc10' or 'esc50'
OUTPUT_CLASSES = 10 if WHICH_DATASET == 'esc10' else 50
N_SAMPLES = 400 if WHICH_DATASET == 'esc10' else 2000

DEFAULT_IMAGE_NAME = 'tmp'
DEFAULT_AUDIO_NAME = 'audio'

METADATA_FILE_PATH = 'meta/esc50.csv'
AUDIO_FOLDER = 'audio/'

SR = 22050 # sample rate
DURATION = 5 # duration in seconds
TOTAL_SAMPLES = SR * DURATION

TRAIN_FOLDS = [1,2,3,4]
VAL_FOLDS = [5]
TEST_FOLDS = [5]

################
### TRAINING ###
################

# data augmentation
WHITE_NOISE_EPSILONS = [] #[0.005,0.01,0.02]
PITCH_SHIFTS = [] #[-2,-1,1,2] # semitones of pitch shifts
TIME_STRETCHES = [] #[0.8,1.2] # rates of time stretches
TIME_DELAYS = 5 # CODED SO THEY HAPPEN POST!!!!

# threshold for silence (depends on preprocessing!!)
# THRESHOLD = -1.0
# discarding if "np.max(segment) - np.min(segment) > 0.001"

# random time delays

#before:
# N_TIME_DELAYS = 10, MIN_LENGTH = 0.01, MAX_LENGTH = 0.15

N_TIME_DELAYS, MIN_LENGTH, MAX_LENGTH = 10, 0.1, 0.3

# log scaled mel spectrograms
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 60

# segmentation
FRAMES_LENGTH = 41
NUMBER_OF_SEGMENTS = 10

# model
DROPOUT = 0.5
WEIGHT_DECAY = 0.001
LEARNING_RATE = 0.002
MOMENTUM = 0.9
OPTIMIZER = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE, momentum=MOMENTUM, nesterov=True)
# OPTIMIZER = tf.keras.optimizers.experimental.Adagrad(learning_rate=LEARNING_RATE, initial_accumulator_value=0.1, epsilon=1e-07, weight_decay=WEIGHT_DECAY, clipnorm=None, clipvalue=None, global_clipnorm=None, use_ema=False, ema_momentum=0.99, ema_overwrite_frequency=None, jit_compile=True, name='Adagrad')

# training
BATCH_SIZE = 1000
EPOCHS = 300
ES_MONITOR = "val_loss" # quantity monitored by early stopping. Reccomended "val_loss" or "val_accuracy"
ES_PATIENCE = 20 # early stopping patience

