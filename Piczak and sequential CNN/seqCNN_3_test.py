from shutil import copy
import tensorflow as tf
import numpy as np

test_X = np.load('test_X.npy')
test_y = np.load('test_y.npy')

model = tf.keras.models.load_model('models/model.h5')

copy('models/training_history.png', 'results_backup_model2/training_history.png')

from sklearn.metrics import classification_report

test_loss, test_acc = model.evaluate(test_X, test_y)

print()

y_pred = np.argmax(model.predict(test_X), axis = 1)
print(y_pred.shape)

test_y = np.argmax(test_y, axis = 1)
print(test_y.shape)
print(classification_report(test_y, y_pred))


# ALL METRICS

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Calculate accuracy, precision, recall, and F1 score
accuracy = accuracy_score(test_y, y_pred)
precision = precision_score(test_y, y_pred, average = 'weighted')
recall = recall_score(test_y, y_pred, average = 'weighted')
f1 = f1_score(test_y, y_pred, average = 'weighted')

print("Accuracy:", round(accuracy,3))
print("Precision:", round(precision,3))
print("Recall:", round(recall,3))
print("F1 Score:", round(f1,3))
print(round(accuracy,3), round(precision,3), round(recall,3), round(f1,3), sep = ', ')