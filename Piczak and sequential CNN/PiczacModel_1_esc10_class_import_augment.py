import numpy as np
import librosa
from parameters import *
import random
from os import listdir
from tqdm import tqdm
import pandas as pd
from utils.utils import *
from utils.data_augmentation import *

# STEP 0: IMPORT DATASET

df = import_dataframe()

# load dataset

train_audio_list = []
train_target_list = []

val_audio_list = []
val_target_list = []

test_audio_list = []
test_target_list = []

for filename,target,fold in tqdm(list(zip(df['filename'], df['target'],df['fold']))[:], desc = 'Importing dataset'):
    audio, _ = librosa.load(AUDIO_FOLDER + filename)
    
    if fold in TRAIN_FOLDS:
        train_audio_list.append(audio)
        train_target_list.append(target)
    if fold in VAL_FOLDS:
        val_audio_list.append(audio)
        val_target_list.append(target)
    if fold in TEST_FOLDS:
        test_audio_list.append(audio)
        test_target_list.append(target)

# compute mean and std before augmenting the data
print('computing mean and variance')
X_train_ = normalize_spectrogram(train_audio_list)
mean, std = np.mean(X_train_), np.std(X_train_)
del X_train_

# STEP 1: DATA AUGMENTATION

# 0 - dog
# 1 - rooster
# 2 - rain
# 3 - sea_waves
# 4 - crackling_fire
# 5 - crying_baby
# 6 - sneezing
# 7 - clock_tick
# 8 - helicopter
# 9 - chainsaw

train_augmented_dataset = []
train_augmented_targets = []
for j,audio in enumerate(tqdm(train_audio_list, desc = 'augmenting dataset')):
    target = train_target_list[j]
    train_augmented_dataset.append(audio)
    train_augmented_targets.append(target)

    multiplied_white_noise_epsilons, pitch_shifts, time_stretches, n_time_delays = data_augment_esc10_target(target)

    # add white noise
    for epsilon in multiplied_white_noise_epsilons:
        train_augmented_dataset.append(add_multiplied_white_noise(audio, epsilon))
        train_augmented_targets.append(train_target_list[j])
        
    # add pitch shift
    for i in pitch_shifts:
        new_audio = add_pitch_shift(audio, semitones = i)
        train_augmented_dataset.append(new_audio)
        train_augmented_targets.append(train_target_list[j])
        
    # add time stretch
    for rate in time_stretches:
        new_audio = add_time_stretch(audio, rate = rate)
        train_augmented_dataset.append(new_audio)
        train_augmented_targets.append(train_target_list[j])


del train_audio_list
del train_target_list

train_augmented_dataset_2 = []
train_augmented_targets_2 = []

for j,audio in enumerate(tqdm(train_augmented_dataset, desc = 'applying random time delays')):
    target = train_augmented_targets[j]
    
    train_augmented_dataset_2.append(audio)
    train_augmented_targets_2.append(target)

    multiplied_white_noise_epsilons, pitch_shifts, time_stretches, n_time_delays = data_augment_esc10_target(target)

    # random time delays
    for _ in range(n_time_delays):
        new_audio = add_random_time_delay(audio)
        train_augmented_dataset_2.append(new_audio)
        train_augmented_targets_2.append(train_augmented_targets[j])

del train_augmented_dataset, train_augmented_targets

# convert to spectrograms
print('computing spectrograms')
X_train = normalize_spectrogram(train_augmented_dataset_2, preserve_input=False)
X_val = normalize_spectrogram(val_audio_list, preserve_input=False)
X_test = normalize_spectrogram(test_audio_list, preserve_input=False)

del train_augmented_dataset_2, val_audio_list, test_audio_list

# then normalize wrt the mean and std everything, and then loop over the matrix
# to extract the segments

# normalize spectrograms
X_train = (X_train - mean) / std
X_val = (X_val - mean) / std
X_test = (X_test - mean) / std

# segment and compute deltas
print('segmenting')
X_train,Y_train= segment_deltas(X_train, train_augmented_targets_2)
X_val,  Y_val  = segment_deltas(X_val, val_target_list)
X_test, Y_test = segment_deltas(X_test, test_target_list)

print('number of targets:',np.unique(Y_train, return_counts = True))

del train_augmented_targets_2, val_target_list, test_target_list

np.save('X_train.npy', X_train)
np.save('Y_train.npy', Y_train)

np.save('X_val.npy', X_val)
np.save('Y_val.npy', Y_val)

np.save('X_test.npy', X_test)
np.save('Y_test.npy', Y_test)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
