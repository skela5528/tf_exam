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
    'validation_split': .9,
    'interpolation': 'bicubic'
}

get_image_dataset_train = partial(image_dataset_from_directory, directory=None, subset='training', **_IMG_DATASET_PARAMS)
get_image_dataset_validation = partial(image_dataset_from_directory, directory=None, subset='validation', **_IMG_DATASET_PARAMS)