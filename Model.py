
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
from keras import backend as K
import autokeras as ak
from PIL import Image 
import numpy as np
import time
import math


NAME = "Melanoma-PennHackathon-Model-{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

tf.random.set_seed(1234)

import pickle

pickle_in = open(r"D:\Github\Melanoma Hackathon\Xtrain.pickle","rb")
X_train = pickle.load(pickle_in)

pickle_in = open(r"D:\Github\Melanoma Hackathon\ytrain.pickle","rb")
y_train = pickle.load(pickle_in)

pickle_in = open(r"D:\Github\Melanoma Hackathon\Xval.pickle","rb")
X_val = pickle.load(pickle_in)

pickle_in = open(r"D:\Github\Melanoma Hackathon\yval.pickle","rb")
y_val = pickle.load(pickle_in)

pickle_in = open(r"D:\Github\Melanoma Hackathon\Xtest.pickle","rb")
X_test = pickle.load(pickle_in)

pickle_in = open(r"D:\Github\Melanoma Hackathon\ytest.pickle","rb")
y_test = pickle.load(pickle_in)


y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


model = Sequential()
model.add(Conv2D(512, (5, 5), activation="relu", input_shape=(224, 224, 3)))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))

model.add(Conv2D(256, (3, 3),activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3),activation='relu'))
#model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))  
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(128))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train
#You can also use generator to use augmentation during training.

EPOCHS = 50
BATCH_SIZE = 8
TRAIN_SIZE = len(X_train)
VAL_SIZE = len(X_val)

compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))

steps_per_epoch = compute_steps_per_epoch(TRAIN_SIZE)
val_steps = compute_steps_per_epoch(VAL_SIZE)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size = BATCH_SIZE,
    validation_data=(X_val, y_val),
    verbose=1, callbacks=[tensorboard],
    steps_per_epoch = steps_per_epoch,
    validation_steps = val_steps)

score = model.evaluate(X_test, y_test)
print('Test accuracy:', score[1])


















BATCH_SIZE = 8
TRAIN_SIZE = len(X_train)
VAL_SIZE = len(X_val)
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))

steps_per_epoch = compute_steps_per_epoch(TRAIN_SIZE)
val_steps = compute_steps_per_epoch(VAL_SIZE)

clf = ak.ImageClassifier(max_trials=25)
clf.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=BATCH_SIZE,
    steps_per_epoch = steps_per_epoch,
    validation_steps = val_steps
)

#Evaluate the classifier on test data
_, acc = clf.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

_, acc = model.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")


#Save the model
model.save('Melanoma_model.h5')


#model.summary easy to view
import pandas as pd
table=pd.DataFrame(columns=["Name","Type","Shape"])
for layer in model.layers:
    table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)

table



















