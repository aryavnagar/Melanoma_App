import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from tqdm import tqdm
import random
import pickle

IMG_SIZE = 96

DATADIR = r"D:\Github\Melanoma Hackathon\Melanoma Images\Data\train_sep"

CATEGORIES = ["NotMelanomaWithAug", "MelanomaWithAug"]

# for category in CATEGORIES:  
#     path = os.path.join(DATADIR,category)  
#     for img in os.listdir(path):  
#         img_array = cv2.imread(os.path.join(path,img)) 
#         plt.imshow(img_array, cmap='gray') 
#         plt.show()  # display!

#         break  # we just want one for now so break
#     break  #...and one more!
    
#Train Data
training_data = []

def create_training_data():
    for category in CATEGORIES:
        
        path = os.path.join(DATADIR,category)  
        class_num = CATEGORIES.index(category) 

        for img in tqdm(os.listdir(path)): 
            try:
                img_array = cv2.imread(os.path.join(path,img)) 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB )
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                training_data.append([new_array, class_num]) 
            except Exception as e: 
                pass


create_training_data()

print(len(training_data))


random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
    
    
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)

imgplot = plt.imshow(X[0])



import pickle

pickle_out = open(r"D:\Github\Melanoma Hackathon\Xtrain.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(r"D:\Github\Melanoma Hackathon\ytrain.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()

##FIle path of pickle
# import os
# print(os.path.abspath("y.pickle"))


#Validation Data
DATADIR = r"D:\Github\Melanoma Hackathon\Melanoma Images\Main\valid"
CATEGORIES = ["NotMelanoma", "Melanoma"]

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img)) 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB )
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))


random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
    
    
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
imgplot = plt.imshow(X[0])


import pickle

pickle_out = open(r"D:\Github\Melanoma Hackathon\Xval.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(r"D:\Github\Melanoma Hackathon\yval.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()





#Test Data
DATADIR = r"D:\Github\Melanoma Hackathon\Melanoma Images\Data\test"
CATEGORIES = ["NotMelanoma", "Melanoma"]

training_data = []

def create_training_data():
    for category in CATEGORIES:  # do dogs and cats

        path = os.path.join(DATADIR,category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)  # get the classification  (0 or a 1). 0=dog 1=cat

        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path,img)) 
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB )
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass
            #except OSError as e:
            #    print("OSErrroBad img most likely", e, os.path.join(path,img))
            #except Exception as e:
            #    print("general exception", e, os.path.join(path,img))

create_training_data()

print(len(training_data))


random.shuffle(training_data)
for sample in training_data[:10]:
    print(sample[1])
    
    
    
X = []
y = []

for features,label in training_data:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 3))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
imgplot = plt.imshow(X[0])


import pickle

pickle_out = open(r"D:\Github\Melanoma Hackathon\Xtest.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open(r"D:\Github\Melanoma Hackathon\ytest.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()



















