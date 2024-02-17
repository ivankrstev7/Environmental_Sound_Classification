
from utils.utils import *
import time

X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')

Y_train = np.load('Y_train.npy')
Y_val = np.load('Y_val.npy')
Y_test = np.load('Y_test.npy')

print(X_test.shape)

# Per-segment performance

model = Piczak_model(input_shape = (60, FRAMES_LENGTH, 2), output_classes = OUTPUT_CLASSES)

model.load_weights('models/model.h5')

evaluation_results = model.evaluate(X_test, Y_test)

# On segments
print("Loss:", round(evaluation_results[0],4))
print("Accuracy:", round(evaluation_results[1],4))


# majority on segments

df = import_dataframe()

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


correct_maj_voting = np.zeros(OUTPUT_CLASSES)
correct_highest_prob = np.zeros(OUTPUT_CLASSES)
total = np.zeros(OUTPUT_CLASSES)

outputs_matrix = np.zeros((OUTPUT_CLASSES, OUTPUT_CLASSES)) # rows are true target, columns are predicted output

y_pred = np.zeros(len(test_target_list))
y_true = np.array(test_target_list)

for i in tqdm(range(len(test_target_list)), desc = 'calculating results on test set'):
    audio = test_audio_list[i]
    target = test_target_list[i]

    segments = normalize_spectrogram([audio], verbose = False)
    segments = (segments - mean) / std
    segments, _ = segment_deltas(segments, [target], verbose = False)

    # shape of "segments": (n_segments, 60, 41, 2)
    out = model.predict(segments, verbose = 0)
    
    # PER SEGMENT MATRIX
    for segment in out:
        outputs_matrix[target][np.argmax(segment)] += 1

    # MAJ VOTING
    per_segment_votes = np.argmax(out, axis = 1)
    maj_voting_out = np.argmax(np.bincount(per_segment_votes))

    # PROBABILITY VOTING
    average_of_probabilities = np.sum(out, axis = 0) / segments.shape[0]
    highest_prob_out = np.argmax(average_of_probabilities)

    # compute accuracy
    correct_maj_voting[target] += (target == maj_voting_out)
    correct_highest_prob[target] += (target == highest_prob_out)
    total[target] += 1

    y_pred[i] = highest_prob_out

d = {}
for target,category in tqdm(list(zip(df['target'], df['category']))):
    d[target] = category

results = ''

results += f'Loss: {round(evaluation_results[0],4)}'
results += '\n'
results += f'Accuracy: {round(evaluation_results[1],4)}'
results += '\n'

results += '\n'
results += f'maj voting correct percentage (total): {np.round(correct_maj_voting.sum() / total.sum() * 100,2)} %'
results += '\n'
results += f'highest prob correct percentage (total): {np.round(correct_highest_prob.sum() / total.sum() * 100,2)} %'
results += '\n'

results += '\n'
results += f'maj voting correct percentage (per-class):'
results += '\n'
for i in range(OUTPUT_CLASSES):
    results += f'{i} - {d[i]} {np.round(correct_maj_voting / total * 100,2)[i]} %'
    results += '\n'

results += '\n'
results += f'highest prob correct percentage (per-class):'
results += '\n'
for i in range(OUTPUT_CLASSES):
    results += f'{i} - {d[i]} {np.round(correct_highest_prob / total * 100,2)[i]} %'
    results += '\n'

print(results)



# SAVE RESULTS

from os import listdir
from shutil import copyfile

backup_folder = 'results_backup/'
number = 1 + int(sorted(listdir(backup_folder))[-1][:3])

threedigitize = lambda x: '0'*(3 - len(str(x))) + str(x)

filename = f'{threedigitize(number)}_results.txt'

parameters = open("parameters.py",'r').read()

with open(backup_folder + filename, 'w') as text:
    text.write(results)
    text.write('\n\n\nPARAMETERS:\n\n')
    text.write(parameters)
    text.write('\n')

copyfile('models/training_history.png', backup_folder + f'{threedigitize(number)}_training_history.png')

# plot matrix
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

labels = ['dog', 'rooster', 'rain', 'sea_waves', 'crackling_fire', 'crying_baby', 'sneezing', 'clock_tick', 'helicopter', 'chainsaw']
cmap = LinearSegmentedColormap.from_list('cmap', [(0, 1, 0), (1, 0, 0)])
plt.imshow(np.eye(10), cmap = cmap)

for (j,i),label in np.ndenumerate(outputs_matrix):
    plt.text(i,j,int(label),ha='center',va='center')
    plt.text(i,j,int(label),ha='center',va='center')

plt.xticks(range(10),labels, rotation = 90)
plt.yticks(range(10),labels)
plt.savefig(backup_folder + f'{threedigitize(number)}_outputs_matrix.png')


# ALL METRICS

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Assuming model.predict_classes(x_test) returns the predicted classes

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average = 'weighted')
recall = recall_score(y_true, y_pred, average = 'weighted')
f1 = f1_score(y_true, y_pred, average = 'weighted')

print("Accuracy:", round(accuracy,3))
print("Precision:", round(precision,3))
print("Recall:", round(recall,3))
print("F1 Score:", round(f1,3))

print(round(accuracy,3), round(precision,3), round(recall,3), round(f1,3), sep = ', ')