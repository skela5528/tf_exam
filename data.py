from functools import partial

import tensorflow as tf
from tensorflow.keras.preprocessing import image, text, sequence
from tensorflow.keras.preprocessing import image_dataset_from_directory


_IMG_DATASET_PARAMS = {
    'labels': 'inferred',
    'label_mode': 'int',
    'batch_size': 32,
    'image_size': (256, 256),
    'seed': 111,
    'validation_split': .95,
    'interpolation': 'bicubic'
}

get_image_dataset_train = partial(image_dataset_from_directory, directory=None, subset='training', **_IMG_DATASET_PARAMS)
get_image_dataset_validation = partial(image_dataset_from_directory, directory=None, subset='validation', **_IMG_DATASET_PARAMS)

# # # DATA GEN # # #
VAL_SPLIT = .2
DATA_DIR = ""
BS = 64
IMG_SIZE = (128, 128)
img_train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=.1,
                                                                    width_shift_range=0.1,
                                                                    height_shift_range=0.1,
                                                                    zoom_range=0.1,
                                                                    horizontal_flip=True,
                                                                    validation_split=VAL_SPLIT,
                                                                    rescale=1/255)
img_validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=VAL_SPLIT, rescale=1/255)

# FROM DIRECTORY ##
# class_mode: One of "categorical", "binary", "sparse"
img_train_datagen.flow_from_directory(directory=DATA_DIR, target_size=IMG_SIZE, batch_size=BS, interpolation='bicubic', class_mode='categorical', seed=111)
img_validation_datagen.flow_from_directory(directory=DATA_DIR, target_size=IMG_SIZE, batch_size=BS, interpolation='bicubic', class_mode='categorical', seed=111)

# FROM NP.ARRAY ##
# img_train_datagen.flow(x, y, batch_size=BS)
# img_validation_datagen.flow(valid_x, valid_y, batch_size=BS)