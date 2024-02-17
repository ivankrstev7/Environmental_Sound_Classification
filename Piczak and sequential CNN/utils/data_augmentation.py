import numpy as np
import random
from parameters import *
import librosa

def add_white_noise(numpy_array, epsilon):
    white_noise = np.random.randn(numpy_array.shape[0]).astype(np.float32) * epsilon
    return np.clip(numpy_array + white_noise, -1, 1)

def add_multiplied_white_noise(numpy_array, epsilon):
    white_noise = np.random.randn(numpy_array.shape[0]).astype(np.float32) * epsilon * np.abs(numpy_array)
    return np.clip(numpy_array + white_noise, -1, 1)

def add_random_time_delay(numpy_array, n_time_delays = N_TIME_DELAYS, min_length = MIN_LENGTH, max_length = MAX_LENGTH):
    """
    n_time_delays - how many time delays
    min_length - minimum length of delay (seconds)
    max_length - maximum length of delays (seconds)
    """
    
    td_array = numpy_array.copy()

    min_samples = int(min_length * SR)
    max_samples = int(max_length * SR)
    positions = random.sample(range(len(numpy_array)),n_time_delays) # randomize positions of time delays
    lengths = [random.randint(min_samples, max_samples) for _ in range(n_time_delays)] # randomize lengths of time delays, from a tenth of a second to half a second

    j = 1
    while j < len(positions):
        for i in range(j):
            positions[j] += lengths[i]
        j += 1

    for i in range(n_time_delays):
        td_array = np.insert(td_array, positions[i], np.zeros(lengths[i]))
    
    # crop to return a clip exactly 5 seconds long
    crop_position = random.randint(0, len(td_array) - TOTAL_SAMPLES)
    td_array = td_array[crop_position:TOTAL_SAMPLES + crop_position]

    return td_array

def add_pitch_shift(numpy_array, semitones):
    # pitch shift audio sample
    data = librosa.effects.pitch_shift(numpy_array, sr = SR, n_steps = semitones)
    return data

def add_time_stretch(numpy_array, rate):
    y = librosa.effects.time_stretch(numpy_array, rate = rate)
    if len(y) > TOTAL_SAMPLES:
        i = random.randint(0, len(y) - TOTAL_SAMPLES)
        data = y[i:i+TOTAL_SAMPLES]
    else:
        missing_samples = TOTAL_SAMPLES - len(y)
        add_left = missing_samples//2
        add_right = missing_samples - add_left
        data = np.concatenate([np.zeros(add_left), y, np.zeros(add_right)]).astype(np.float32)
    assert len(numpy_array) == len(data), "something went wrong here!"
    return data

def spec_augment(spec: np.ndarray, num_mask=2, 
                 freq_masking_max_percentage=0.15, time_masking_max_percentage=0.3):

    spec = spec.copy()
    for i in range(num_mask):
        all_frames_num, all_freqs_num = spec.shape
        freq_percentage = random.uniform(0.0, freq_masking_max_percentage)
        
        num_freqs_to_mask = int(freq_percentage * all_freqs_num)
        f0 = np.random.uniform(low=0.0, high=all_freqs_num - num_freqs_to_mask)
        f0 = int(f0)
        spec[:, f0:f0 + num_freqs_to_mask] = 0

        time_percentage = random.uniform(0.0, time_masking_max_percentage)
        
        num_frames_to_mask = int(time_percentage * all_frames_num)
        t0 = np.random.uniform(low=0.0, high=all_frames_num - num_frames_to_mask)
        t0 = int(t0)
        spec[t0:t0 + num_frames_to_mask, :] = 0
    
    return spec

def add_random_silence(numpy_array):
    None

def data_augment_esc10_target(target):
    """
    returns specific data augmentation parameters for the target
    """
    if target in [0,1]: # dog, rooster
        multiplied_white_noise_epsilons = [0.05,0.1]
        pitch_shifts = [-2,2]
        time_stretches = [0.9,1.1]
        n_time_delays = 3

    if target in [2,3]: # rain, sea_waves
        multiplied_white_noise_epsilons = []
        pitch_shifts = []
        time_stretches = [0.9,1.2,1.5,2.0]
        n_time_delays = 5
    
    if target in [4]: # crackling_fire
        multiplied_white_noise_epsilons = [0.05,0.1]
        pitch_shifts = []
        time_stretches = []
        n_time_delays = 6

    if target in [5]: # crying_baby
        multiplied_white_noise_epsilons = [0.1]
        pitch_shifts = [-1,1]
        time_stretches = [0.9,1.1]
        n_time_delays = 4
    
    if target in [6]: # sneezing
        multiplied_white_noise_epsilons = [0.05,0.1, 0.2, 0.5]
        pitch_shifts = [-4,-3,-2,-1,1,2,3,4]
        time_stretches = [0.5,0.7,1.5,2.0,3.0, 4.0]
        n_time_delays = 3
    
    if target in [7]: # clock_tick
        multiplied_white_noise_epsilons = [0.05,0.1]
        pitch_shifts = []
        time_stretches = []
        n_time_delays = 4
    
    if target in [8,9]: # helicopter, chainsaw
        multiplied_white_noise_epsilons = [0.01,0.05]
        pitch_shifts = []
        time_stretches = [0.8,1.2,1.5]
        n_time_delays = 5
    
    return multiplied_white_noise_epsilons, pitch_shifts, time_stretches, n_time_delays