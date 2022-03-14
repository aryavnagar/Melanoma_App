import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, MaxPool2D, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras import backend as K
from tensorflow import keras
import autokeras as ak
from PIL import Image 
import numpy as np
import time
import math

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


NAME = "Melanoma-Hackathon-Model{}".format(int(time.time()))
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


BATCH_SIZE = 64
TRAIN_SIZE = len(X_train)
VAL_SIZE = len(X_val)
compute_steps_per_epoch = lambda x: int(math.ceil(1. * x / BATCH_SIZE))

steps_per_epoch = compute_steps_per_epoch(TRAIN_SIZE)
val_steps = compute_steps_per_epoch(VAL_SIZE)

# AutoKeras
# clf = ak.ImageClassifier(max_trials=25)
# clf.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=10,
#     batch_size=BATCH_SIZE,
#     steps_per_epoch = steps_per_epoch,
#     validation_steps = val_steps)

# #Evaluate the classifier on test data
# _, acc = clf.evaluate(X_test, y_test)
# print("Accuracy = ", (acc * 100.0), "%")

# # get the final best performing model
# model = clf.export_model()
# print(model.summary())

# _, acc = model.evaluate(X_test, y_test)
# print("Accuracy = ", (acc * 100.0), "%")


# #Save the model
# model.save('Melanoma_model.h5')

# # Load the model
# model = keras.models.load_model(r"D:\Github\Melanoma Hackathon\Best Models By Date\weights-improvement-02-0.92.hdf5")

# #Model Checkpoint
# save_filepath = r"D:\Github\Melanoma Hackathon\Best Models By Date\Saved_Models\weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
# checkpoint = ModelCheckpoint(save_filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# # Early Stopping
# early_stop = EarlyStopping(monitor='val_loss', patience=6, verbose=1)

# #CSV logger (logs epoch, accuracy, val_accuracy, and val_loss)
# log_filepath = r"D:\Github\Melanoma Hackathon\Best Models By Date\model_logs.csv"
# log_csv = CSVLogger('log_filepath',separator=',', append=False)
    
# model.fit(
#     X_train,
#     y_train,
#     validation_data=(X_val, y_val),
#     epochs=50,
#     batch_size=BATCH_SIZE,
#     steps_per_epoch = steps_per_epoch,
#     validation_steps = val_steps,
#     callbacks=[tensorboard, checkpoint, log_csv, early_stop]
# )

# model.save()


model = keras.models.load_model(r"D:\Github\Melanoma Hackathon\Best Models By Date\weights-improvement-02-0.92.hdf5")
#model.summary easy to view
import pandas as pd
table=pd.DataFrame(columns=["Name","Type","Shape"])
for layer in model.layers:
    table = table.append({"Name":layer.name, "Type": layer.__class__.__name__,"Shape":layer.output_shape}, ignore_index=True)

table
















