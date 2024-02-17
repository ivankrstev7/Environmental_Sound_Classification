from parameters import *
from utils.utils import *
import numpy as np

train_X = np.load('train_X.npy')
train_y = np.load('train_y.npy')
test_X = np.load('test_X.npy')
test_y = np.load('test_y.npy')

print(train_X.shape)
print(train_y.shape)

DROPOUT_MODEL2 = 0.0
L2_COEFF = 0.0
model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(32, (3, 3), dilation_rate=(1,1), input_shape=train_X.shape[1:], kernel_regularizer=tf.keras.regularizers.l2(L2_COEFF)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(DROPOUT_MODEL2))

model.add(tf.keras.layers.Conv2D(64, (3, 3), dilation_rate=(1,1), kernel_regularizer=tf.keras.regularizers.l2(L2_COEFF)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(DROPOUT_MODEL2))

model.add(tf.keras.layers.Conv2D(128, (3, 3),dilation_rate=(1,1) , kernel_regularizer=tf.keras.regularizers.l2(L2_COEFF)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(DROPOUT_MODEL2))

model.add(tf.keras.layers.Conv2D(256, (3, 3),dilation_rate=(1,1) , kernel_regularizer=tf.keras.regularizers.l2(L2_COEFF)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(DROPOUT_MODEL2))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)))  
model.add(tf.keras.layers.Activation('relu'))
model.add(tf.keras.layers.Dropout(DROPOUT_MODEL2))

#--- Softmax classifier
#
model.add(tf.keras.layers.Dense(OUTPUT_CLASSES))
model.add(tf.keras.layers.Activation('softmax'))
#------ Otimizers
opt=tf.keras.optimizers.Adam(
    learning_rate=0.0001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-08,
    amsgrad=True,
    clipnorm=None,
    clipvalue=None,
    global_clipnorm=None,
    name='Adam')

sgd = tf.keras.optimizers.SGD(learning_rate=0.00025, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
            #   optimizer=sgd,
              optimizer=opt,
              metrics=['accuracy'])

accuracy_threshold=0.92

model.summary()

class myCallback(tf.keras.callbacks.Callback): 
    def on_epoch_end(self, epoch, logs={}): 
        if(logs.get('val_accuracy') > accuracy_threshold):   
            print("\nReached %2.2f%% accuracy, we stop training." %(accuracy_threshold*100))   
            self.model.stop_training = True

custom_early_stopping = myCallback()

history = model.fit(
    train_X, 
    train_y, 
    epochs=50,
    steps_per_epoch=len(train_X)//16,
    #validation_split=0.2, 
    validation_data=(test_X, test_y),
    batch_size=10, 
    #verbose=2,
    callbacks=[custom_early_stopping]
)

save_results('models/', model, history)
