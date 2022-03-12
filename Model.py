import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import autokeras as ak
from PIL import Image 
import numpy as np

tf.random.set_seed(1234)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)


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


clf = ak.ImageClassifier(max_trials=25)
clf.fit(
    X_train,
    y_train,
    # Use your own validation set.
    validation_data=(X_val, y_val),
    epochs=10,
)

#Evaluate the classifier on test data
_, acc = clf.evaluate(X_test, y_test)
print("Accuracy = ", (acc * 100.0), "%")

# get the final best performing model
model = clf.export_model()
print(model.summary())

#Save the model
model.save('cifar_model.h5')
