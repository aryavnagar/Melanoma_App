import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
from scipy import stats
import shutil
import PIL
import random



# ISIC 2019

ISIC_2019_truth = pd.read_csv(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\ISIC_2019_Training_GroundTruth.csv")
ISIC_2019_metadata = pd.read_csv(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\ISIC_2019_Training_Metadata.csv")

metadata = [ISIC_2019_truth.drop(['UNK'], axis=1), ISIC_2019_metadata.drop(['image'], axis=1)]

ISIC_2019csv = pd.concat(metadata, axis=1)


#Data Visualization
fig = plt.figure(figsize=(20,15))

ax1 = fig.add_subplot(221)
Class = ['MEL','NV','BCC','AK','BKL','DF','VASC','SCC']
no = [ISIC_2019csv['MEL'].value_counts()[0],ISIC_2019csv['NV'].value_counts()[0]
      ,ISIC_2019csv['BCC'].value_counts()[0],ISIC_2019csv['AK'].value_counts()[0]
      ,ISIC_2019csv['BKL'].value_counts()[0],ISIC_2019csv['DF'].value_counts()[0]
      ,ISIC_2019csv['VASC'].value_counts()[0],ISIC_2019csv['SCC'].value_counts()[0]]

yes = [ISIC_2019csv['MEL'].value_counts()[1],ISIC_2019csv['NV'].value_counts()[1]
      ,ISIC_2019csv['BCC'].value_counts()[1],ISIC_2019csv['AK'].value_counts()[1]
      ,ISIC_2019csv['BKL'].value_counts()[1],ISIC_2019csv['DF'].value_counts()[1]
      ,ISIC_2019csv['VASC'].value_counts()[1],ISIC_2019csv['SCC'].value_counts()[1]]


legend = ['No','Yes']
w = 0.6

ax1.bar(Class, no, w)
ax1.bar(Class, yes, w, bottom=no)
ax1.legend(legend, loc=2)

ax2 = fig.add_subplot(222)
ISIC_2019csv['sex'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_ylabel('Count', size=15)
ax2.set_title('Sex');

ax3 = fig.add_subplot(223)
ISIC_2019csv['anatom_site_general'].value_counts().plot(kind='bar')
ax3.set_ylabel('Count',size=12)
ax3.set_title('Localization')


ax4 = fig.add_subplot(224)
sample_age = ISIC_2019csv[pd.notnull(ISIC_2019csv['age_approx'])]
sns.distplot(sample_age['age_approx'], fit=stats.norm, color='red');
ax4.set_title('Age')

plt.tight_layout()
plt.show()

#Move images to different folder 
count = 0
melFolder = r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\melanoma"
notmelFolder = r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\notmelanoma"
for x in ISIC_2019csv['MEL']:
    
    if (ISIC_2019csv['MEL'][count] == 0):
        path = os.path.join(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\ISIC_2019_Training_Input\ISIC_2019_Training_Input",  str(ISIC_2019csv['image'][count]) + '.jpg')
        shutil.move(path, notmelFolder)

    elif (ISIC_2019csv['MEL'][count] == 1):
        path = os.path.join(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\ISIC_2019_Training_Input\ISIC_2019_Training_Input",  str(ISIC_2019csv['image'][count]) + '.jpg')
        shutil.move(path, melFolder)
        
    count = count +1

path = os.path.join(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2019\ISIC_2019_Training_Input\ISIC_2019_Training_Input",  str(ISIC_2019csv['image'][0]) + '.jpg')
Image.open(open(path, 'rb'))


ISIC_2020csv = pd.read_csv(r'D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2020\ISIC_2020_Training_GroundTruth.csv')

#Move images to different folder 
count = 0
melFolder = r"D:\Github\Melanoma Hackathon\Melanoma Images\melanoma"
notmelFolder = r"D:\Github\Melanoma Hackathon\Melanoma Images\notmelanoma"

for x in ISIC_2020csv['target']:
    
    if (ISIC_2020csv['target'][count] == 0):
        path = os.path.join(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2020\ISIC_2020_Training_JPEG\train",  str(ISIC_2020csv['image_name'][count]) + '.jpg')
        shutil.move(path, notmelFolder)

    elif (ISIC_2020csv['target'][count] == 1):
        path = os.path.join(r"D:\Github\Melanoma Hackathon\Melanoma Images\ISIC 2020\ISIC_2020_Training_JPEG\train",  str(ISIC_2020csv['image_name'][count]) + '.jpg')
        shutil.move(path, melFolder)
        
    count = count +1


#Remove dark edges of image
folder_dir = r"D:\Github\Melanoma Hackathon\Melanoma Images\notmelanoma"
for images in os.listdir(folder_dir):
    
    path = os.path.join(folder_dir,  str(images))
    red_image = PIL.Image.open(path)
    red_image_rgb = red_image.convert("RGB")
    rgb_pixel_value = red_image_rgb.getpixel((0,0))
    
    if ((rgb_pixel_value) < (40, 40, 40)):
        os.remove(path)
    elif ((rgb_pixel_value) > (200,200,200)):
        os.remove(path)






red_image = PIL.Image.open(r"D:\Github\Melanoma Hackathon\Melanoma Images\melanoma\ISIC_0000030_downsampled.jpg")
red_image_rgb = red_image.convert("RGB")
rgb_pixel_value = red_image_rgb.getpixel((0,0))
print(rgb_pixel_value)

if (str(rgb_pixel_value) == "(0, 0, 0)"):
    os.remove(path)  

if ((0,0,0) < (10,10,10)):
    print(True)

#Test plot coords on image (where are we checking for dark pixles)
im = plt.imread(r"D:\Github\Melanoma Hackathon\Melanoma Images\melanoma\ISIC_0000030_downsampled.jpg")
implot = plt.imshow(im)
x=[0]
y=[0]
for p,q in zip(x,y):
    x_cord = p # try this change (p and q are already the coordinates)
    y_cord = q
    plt.scatter([x_cord], [y_cord])
plt.show()

#Randomly move x amount of images from one folder to another to maintain balance between yes and no melanoma
source = r"D:\Github\Melanoma Hackathon\Melanoma Images\notmelanoma"
dest = r"D:\Github\Melanoma Hackathon\Melanoma Images\DermMel\train_sep\fdsh"
files = os.listdir(source)
no_of_files = 133

for file_name in random.sample(files, no_of_files):
    shutil.move(os.path.join(source, file_name), dest)




