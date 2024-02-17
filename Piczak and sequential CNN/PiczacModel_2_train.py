import numpy as np
import librosa
from parameters import *
import random
from os import listdir
from tqdm import tqdm
import pandas as pd
from utils.utils import *
from utils.data_augmentation import *

X_train = np.load('X_train.npy')
Y_train = np.load('Y_train.npy')

X_val = np.load('X_val.npy')
Y_val = np.load('Y_val.npy')

X_test = np.load('X_test.npy')
Y_test = np.load('Y_test.npy')

# STEP 5: DEFINE MODEL

model = Piczak_model(input_shape = (60, FRAMES_LENGTH, 2), output_classes = OUTPUT_CLASSES)

model.summary()

total_params = model.count_params()
bytes_per_param = 4
model_size_kb = total_params * bytes_per_param / 10**6
print(f'Estimated model size: {model_size_kb:.2f} MB')

# training

checkpoint_filename = "models/model.h5" #"models/model_epoch_{epoch:02d}.h5"

checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_filename, save_weights_only=True)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor=ES_MONITOR, min_delta=0, patience=ES_PATIENCE, restore_best_weights=False)

import time

t0 = time.time()

history = model.fit(X_train,
          Y_train,
          validation_data=(X_val, Y_val),
          batch_size = BATCH_SIZE,
          epochs = EPOCHS,
          callbacks = [early_stopping],
          shuffle = True,
          )

print("Training time: ",time.time() - t0, "s")

model.save_weights(checkpoint_filename)

# model.evaluate(X_test, Y_test)

# Access loss values from the training history
train_loss = history.history['loss']
val_loss = history.history['val_loss']  # if you have validation data

# Save the loss values to a file or process them as needed
with open('models/train_loss.txt', 'w') as file:
    for loss in train_loss:
        file.write(f'{loss}\n')

with open('models/val_loss.txt', 'w') as file:
    for loss in val_loss:
        file.write(f'{loss}\n')

import matplotlib.pyplot as plt

# Plot the loss curve
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('models/training_history.png')
