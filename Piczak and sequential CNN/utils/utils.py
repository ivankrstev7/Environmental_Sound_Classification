import numpy as np
import matplotlib.pyplot as plt
from parameters import *
from scipy.io.wavfile import write
import tensorflow as tf
from tqdm import tqdm
import librosa
import pandas as pd

def plot_waveform(y, normalize = False, number = 0):
    f"""
    saves plot showing the waveform y (numpy array).
    The plot is saved as "{DEFAULT_IMAGE_NAME}{number}.png"
    :y: numpy array representing the audio file (output of librosa.load);
    :normalize: whether you want to show the normalized waveform (rescaled to [-1,1]);
    :number: the number in the saved file, change it to save to different files.
    """
    if normalize:
        max_ = np.max(np.abs(y))
        y = y / max_
    plt.plot(y)
    plt.title('Signal')
    plt.xlabel('Time(samples)')
    plt.ylabel('Amplitude')
    plt.ylim([-1,1])
    plt.savefig(DEFAULT_IMAGE_NAME + str(number) + '.png')
    plt.close()

def save_as_wav(y, number = 0):
    f"""
    saves .wav file containing the sound encoded in y (numpy array).
    The sound is saved as "{DEFAULT_AUDIO_NAME}{number}.wav"
    """
    write(f'{DEFAULT_AUDIO_NAME}{number}.wav',SR, y)

def plot_matrix(y, number = 0):
    """
    simple plt.imshow. Can be used to display spectrograms.
    """
    plt.imshow(y)
    plt.savefig(DEFAULT_IMAGE_NAME + str(number) + '.png')

def normalize_spectrogram(list_dataset, preserve_input = True, verbose = True):
    """
    normalize, then convert to log-scaled mel-spectrogram.
    Returns numpy array with mel spectrograms.
    :list_dataset: a list containing the audio files as numpy arrays.

    ouput: a numpy array shaped (n_samples, _, _), where _,_ are the rows and the columns of the spectrograms.
    """

    # normalize
    if preserve_input:
        normalized_dataset = list_dataset.copy()
    else:
        normalized_dataset = list_dataset

    for i in tqdm(range(len(normalized_dataset)), desc = 'normalizing audio files', disable = not verbose):
        normalized_dataset[i] = librosa.util.normalize(normalized_dataset[i]) 

    # STEP 2: EXTRACT LOG-SCALED MEL-SPECTROGRAMS

    mel_spectrograms = normalized_dataset

    for i in tqdm(range(len(mel_spectrograms)), desc = 'extracting log scaled mel spectrograms', disable = not verbose):
        audio = mel_spectrograms[i]
        melspec = librosa.feature.melspectrogram(   y = audio,
                                                    sr = SR, 
                                                    n_fft = N_FFT, 
                                                    hop_length = HOP_LENGTH,
                                                    n_mels = N_MELS)
        # melspec_log = np.log(1 + melspec)
        melspec_log= librosa.power_to_db(melspec, ref=np.max)
        mel_spectrograms[i] = melspec_log
    
    output = np.stack(mel_spectrograms)

    return output

# you can ignore this function
def return_segments_indexes(total, partial, number_of_segments, print_percentage_of_overlap = False):
    """
    Given a sequence of integers [0, 1, 2, ..., total-1]
    will give the indexes to split in sub-sequences [0,1,2,...,partial], [x,x+1,...,x+partial]
    in such a way that they are covering evenly the full sequence, minimizing the overlap.
    The total number of sub-sequences is number_of_segments.
    Set print_percentage_of_overlap to check the average overlap between adjacent regions
    """
    half_left = partial // 2
    half_right = partial - half_left

    centers = [round(half_left + i*(total - partial) / (number_of_segments-1)) for i in range(number_of_segments)]
    edges = [(center - half_left, center + half_right) for center in centers]
    if print_percentage_of_overlap:
        tmp = [edges[i][1] - edges[i+1][0] for i in range(number_of_segments - 1)]
        print(f'percentage of overlap: {round(sum(tmp)/len(tmp) / partial * 100, 1)} %')
    return edges

def segment_deltas(X, list_targets, verbose = True): # X shape: (n_samples, n_mels, total_frames) --> (n_samples, 60, 41)
    """
    :X: numpy array of shape (n_samples, _, _) where _, _ are rows and cols of spectrograms.
    :list_targets: a list of length (n_samples) containing the targets.

    """
    # STEP 3: SPLIT INTO OVERLAPPING SEGMENTS

    total_segments = X.shape[2]
    frames_length = FRAMES_LENGTH
    number_of_segments = NUMBER_OF_SEGMENTS

    edges = return_segments_indexes(total_segments, frames_length, number_of_segments)

    segments = []
    segments_targets = []

    discarded_counter = 0
    total_counter = 0

    for j,melspec in enumerate(tqdm(X, desc = 'segmenting spectrograms', disable = not verbose)):
        for left_edge, right_edge in edges:
            segment = melspec[:,left_edge:right_edge].copy()

            # threshold = THRESHOLD
            # if np.max(segment) > threshold:

            # discard silent segments
            if np.max(segment) - np.min(segment) > 0.001:
                segments.append(segment)
                segments_targets.append(list_targets[j])
            else:
                discarded_counter += 1
            total_counter += 1

    if verbose:
        print(f'Discarded {discarded_counter} silent segments out of {total_counter} total segments')

    # STEP 4: COMPUTE SEGMENTS' DELTAS

    segments_and_deltas = []

    for segment in tqdm(segments, desc = 'computing deltas', disable = not verbose):
        segment_delta = librosa.feature.delta(segment, axis = 1)
        two_channel_input = np.stack((segment, segment_delta), axis = -1)
        segments_and_deltas.append(two_channel_input)

    X = np.stack(segments_and_deltas)
    Y = np.array(segments_targets)

    return X,Y




def Piczak_model(input_shape, output_classes):
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # first convolutional layer
    model.add(tf.keras.layers.Conv2D(80, (57,6), (1,1), padding = 'valid', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)))

    # first max pooling
    model.add(tf.keras.layers.MaxPool2D((4,3),(1,3), padding = 'valid'))
    model.add(tf.keras.layers.Dropout(DROPOUT))

    # second convolutional layer
    model.add(tf.keras.layers.Conv2D(80,(1,3),(1,1), padding = 'same', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)))

    # second max pooling
    model.add(tf.keras.layers.MaxPool2D((1,3),(1,3), padding = 'valid'))

    # flatten
    model.add(tf.keras.layers.Flatten())

    # double dense layer
    model.add(tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    model.add(tf.keras.layers.Dense(5000, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)))
    model.add(tf.keras.layers.Dropout(DROPOUT))
    
    # return result
    model.add(tf.keras.layers.Dense(output_classes, activation = 'softmax', kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY)))

    # set optimizer
    optimizer = OPTIMIZER

    model.compile(optimizer = optimizer,
                loss = 'sparse_categorical_crossentropy',
                metrics = ['accuracy'])
    return model

def import_dataframe():

    # import .csv file
    df = pd.read_csv(METADATA_FILE_PATH)

    if WHICH_DATASET == 'esc10':
        # select only esc10 samples
        df = df[df['esc10']]
        
        # replace the original target values with numbers ranging from 0 to 9
        old_values =[ 0,  1, 10, 11, 12, 20, 21, 38, 40, 41]
        remap_values = list(zip(old_values, range(10)))
        for old, new in remap_values:
            df['target'].replace(old, new, inplace = True)
    
    return df



def return_segments_indexes(total, partial, number_of_segments, print_percentage_of_overlap = False):
    """
    Given a sequence of integers [0, 1, 2, ..., total-1]
    will give the indexes to split in sub-sequences [0,1,2,...,partial], [x,x+1,...,x+partial]
    in such a way that they are covering evenly the full sequence, minimizing the overlap.
    The total number of sub-sequences is number_of_segments.
    Set print_percentage_of_overlap to check the average overlap between adjacent regions
    """
    half_left = partial // 2
    half_right = partial - half_left

    centers = [round(half_left + i*(total - partial) / (number_of_segments-1)) for i in range(number_of_segments)]
    edges = [(center - half_left, center + half_right) for center in centers]
    if print_percentage_of_overlap:
        tmp = [edges[i][1] - edges[i+1][0] for i in range(number_of_segments - 1)]
        print(f'percentage of overlap: {round(sum(tmp)/len(tmp) / partial * 100, 1)} %')
    return edges

def audio_to_melspec_log(numpy_array):
    """
    normalize, then convert to spectrogram,
    then split into segments, then compute deltas.
    Returns numpy arrays with data and targets
    """

    # normalize

    numpy_array = librosa.util.normalize(numpy_array)

    # STEP 2: EXTRACT LOG-SCALED MEL-SPECTROGRAMS

    melspec = librosa.feature.melspectrogram(   y = numpy_array,
                                                sr = SR, 
                                                n_fft = N_FFT, 
                                                hop_length = HOP_LENGTH,
                                                n_mels = N_MELS)
    melspec_log = np.log(1 + melspec)

    return melspec_log

import random

def return_random_sample_from_category(category_or_target):
    df = pd.read_csv(METADATA_FILE_PATH)
    if isinstance(category_or_target, str):
        df = df[df['category'] == category_or_target]
    elif isinstance(category_or_target, int):
        df = df[df['target'] == category_or_target]
    filename = random.choice(list(df['filename']))
    audio, _ = librosa.load(AUDIO_FOLDER + filename)
    return audio



def save_results(backup_folder, model, history):
    model.save(backup_folder + 'model.h5')

    # Access loss values from the training history
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']  # if you have validation data

    # Save the loss values to a file or process them as needed
    with open(backup_folder  + 'train_loss.txt', 'w') as file:
        for loss in train_loss:
            file.write(f'{loss}\n')

    with open(backup_folder + 'val_loss.txt', 'w') as file:
        for loss in val_loss:
            file.write(f'{loss}\n')

    # Plot the loss curve
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(backup_folder + 'training_history.png')
    plt.close()
