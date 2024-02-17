# https://www.google.com/search?client=firefox-b-d&q=piczak+2015+tensorflow
# https://github.com/DrStef/Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification
# https://github.com/DrStef/Deep-Learning-and-Digital-Signal-Processing-for-Environmental-Sound-Classification/blob/main/ESC10-Sound-Classification-Mel-Spectrograms_v04.ipynb

import tqdm
from parameters import *
import librosa
from utils.utils import *
from utils.data_augmentation import *

audio_list = []
target_list = []

df = import_dataframe()

data = []


for filename,target in tqdm(list(zip(df['filename'], df['target'])), desc = 'Importing dataset'):
        y, fs = librosa.load(AUDIO_FOLDER + filename, sr=SR)
        data.append((y,target))

audio_data = []
labels=[]

for i,j in data:
    audio_data.append(i)
    labels.append(j)

audio_data=np.array(audio_data)
labels=np.array(labels)

import keras
ylabels=keras.utils.to_categorical(labels, num_classes=OUTPUT_CLASSES, dtype='float32')

from skimage import util

sub_sequence= SR*1.25    #  1.25 seconds of signal ! 
st=400 #  samples for sliding the window ith overlap  
audio_data_red = []

# data augmentation
# augmentations = []
# for audio in tqdm(audio_data.copy(), desc = 'augmenting dataset'):
#     new_audio = np.zeros(110250)
#     while np.all(np.abs(new_audio) < 0.01):
#         new_audio = add_random_time_delay(audio, 10, 0.05,0.15)
#     augmentations.append(new_audio)
# audio_data = np.vstack([audio_data,np.stack(augmentations)])

for i in tqdm(range(0,len(audio_data)), desc = 'data reduction'):
    frames = util.view_as_windows(audio_data[i], window_shape=(sub_sequence,), step=st)
    # optim_frame_index= np.dot(frames,frames.T).diagonal().argmax()
    frame_intensity = []
    for frame in frames:
        frame_intensity.append(frame @ frame)
    optim_frame_index = np.array(frame_intensity).argmax()
    # print(optim_frame_index == optim_frame_index_2, optim_frame_index, optim_frame_index_2)
    audio_data_red.append(frames[optim_frame_index]/np.max(frames[optim_frame_index]))   

melspectrogram = []
hp_l= 108  # creates 256 samples in the time domain.  
n_m = 256
NFFT=1024*2  # high definition. Remember 1 sec audio signal <--->  fs= 44100 points  or  fs= 22050 points 
trunc_mel= 256  # number of mels filters 

for audio in tqdm(audio_data_red, desc = 'computing log scaled mel spectrograms'):
    mel_feat = librosa.feature.melspectrogram(y=audio,sr=SR,
                                          n_fft= NFFT, 
                                          hop_length= hp_l, 
                                          win_length= NFFT, 
                                          window='hann', 
                                          center=True, 
                                          power=2, pad_mode='constant', n_mels=n_m)
    
    mel_feat=mel_feat[0:trunc_mel,:]  # Truncation number of mel filters 
    pwr = librosa.power_to_db(mel_feat, ref=1e-3)

    pwr=pwr.reshape(-1,1)
    melspectrogram.append(pwr)

melspectrogram = np.array(melspectrogram ) # 400x65536x1

from sklearn import preprocessing

melspectro=melspectrogram.reshape(N_SAMPLES,-1) # 400x65536
transform = preprocessing.StandardScaler()
normalized_melspectro= transform.fit_transform(melspectro)

features_CNN = np.reshape(normalized_melspectro,(N_SAMPLES,n_m, -1,1)) # 400x256x256x1

from sklearn.model_selection import train_test_split

for i in range(OUTPUT_CLASSES):
    spctrgrm = features_CNN[i,:,:,0]

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

(train_X,test_X,train_y,test_y)= train_test_split(features_CNN, ylabels, test_size=0.2, stratify=ylabels, random_state=5)

print(np.shape(train_X), np.shape(train_y), np.shape(test_X), np.shape(test_y))

np.save('train_X.npy',train_X)
np.save('train_y.npy',train_y)
np.save('test_X.npy',test_X)
np.save('test_y.npy',test_y)
