from keras.preprocessing.image import ImageDataGenerator
from skimage import io

datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        horizontal_flip=True,
        fill_mode='reflect')


i = 0
for batch in datagen.flow_from_directory(directory=r"D:\Github\Melanoma Hackathon\Melanoma Images\Main\train_sep\NotMelanoma", 
                                         batch_size=16,  
                                         target_size=(224, 224),
                                         color_mode="rgb",
                                         save_to_dir=r"D:\Github\Melanoma Hackathon\Melanoma Images\Main\train_sep\aug_notmel", 
                                         save_prefix='aug', 
                                         save_format='jpg'):
    i += 1
    if i > 31:
        break 