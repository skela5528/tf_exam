import json
import pathlib
import numpy as np
import logging
from datetime import datetime
from matplotlib import pyplot as plt

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import backend as K

from net_bloks import *

"""
- The exam contains 5 problems to solve, part of the code is already written and you need to complete it.

- I don't have GPU on my laptop and also lost time while waiting for training to be done 
(never more than ~10 mins each time but it adds up), so if you can get GPU go for it!

- for multiple questions the source comes from TensorFlow Datasets, 
spend some time understanding the structure of the objects you get as a result from load_data

- You will get 5 different files, one per question

1. Use EarlyStopping() keras callback (with restore_best_weights=True) to stop training before overfitting while 
reserving best weights so far.
2. Use ModelCheckpoint() keras callback (with save_best_only=True) to save a copy of your model whenever it gets better.
3. Use include_optimizer=False option in your keras.models.save_model (or model.save) statement, to reduce the size of 
your model. 
Reduce my model’s size from ~300 MB to ~ 41 MB!
4. find best acccuracy -> with same accuracy using fewer/simpler layers, faster optimizer 
(e.g. “RMSprop” is heavier and slower than “Adam”), and/or fewer epochs.

FROM CANDIDATE HANDBOOK
 - Preprocess data to get it ready for use in a model
 - Use models to predict results.
 - augmentation and dropout.
 - Use pretrained models (transfer learning)
 - Extract features from pre-trained models.
 - Use callbacks to trigger the end of training cycles
 - Use datasets in different formats, including json and csv.
 - Use ImageDataGenerator
 - Use RNNS, LSTMs, GRUs and CNNs in models that work with text.
 - Train LSTMs on existing text to generate text (such as songs and poetry)
 - Use TensorFlow for time series forecasting.
 - 

"""
# TODO 1 - prepare sample models for text / for time seq
# TODO 2 - train a model for each topic - take data from 'tfds'
# TODO 3 practice save/ load/ transfer_learning/
# TODO 4 explore tf.data.Dataset interface
# TODO 5 transfer learning

_DATA_DIR = "/home/cortica/.keras/datasets/flower_photos"
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
LOG = logging.getLogger("tf_exam")


def check_environment():
    import sys

    print(f"* check environment *")
    print(f" - python: {sys.version}")
    print(f" - tf version: {tf.__version__}")
    print(f" - tfds version: {tfds.__version__}")
    print(f" - tf is_built_with_cuda: {tf.test.is_built_with_cuda()}")
    tf.config.list_physical_devices('GPU')

    print()


def download_data():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lr = float(K.get_value(self.model.optimizer.lr))
        print(f"\nEnd epoch {epoch + 1}| LR={lr: 0.08f}\n\n")


class LRSchedulePerBatch(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        epoch = len(self.model.history.epoch)
        batch_id = epoch * self.params.get('steps', None) + batch
        new_lr = 1e-8 * 10 ** (batch_id / 20)
        K.set_value(self.model.optimizer.lr, K.get_value(new_lr))
        print(f"\n...Training: end of batch {batch} LR->{new_lr: 0.09f}\n\n")


class TrainingUtils:
    @staticmethod
    def save_history(model_history, out_path=""):
        time_now = datetime.now().strftime("%H:%M:%S")
        out_path += f"hist_{time_now}.json"
        history_dict = model_history.history
        for k, v in history_dict.items():
            if isinstance(v, list):
                history_dict[k] = [float(x) for x in v]
        with open(out_path, 'w') as out_stream:
            json.dump(history_dict, out_stream)

    @staticmethod
    def train_model(model, train_data, validation_data, optimizer, loss='categorical_crossentropy', epochs=3, verbose=1):
        # init
        K.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)

        # optimizer
        opt = tf.keras.optimizers.Adam() if optimizer is None else optimizer

        # compile
        model.compile(opt, loss=loss, metrics=["acc"])

        # fit
        history = model.fit(train_data,
                            validation_data=validation_data,
                            epochs=epochs,
                            verbose=verbose)
        return history

    @staticmethod
    def select_lr(model, train_data, loss='categorical_crossentropy', optimizer=None, epochs=150, batch_size=32,
                  mode="epoch"):
        assert mode in ["epoch", "batch", "epoch_coarse"], LOG.error(f"wrong mode: {mode}!!!")
        initial_lr = 1e-8
        K.clear_session()
        if optimizer is None:
            optimizer = tf.keras.optimizers.SGD(momentum=0.9)
            optimizer._set_hyper("learning_rate", initial_lr)

        lr_schedule = None
        if mode == "epoch":
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
        elif mode == "batch":
            lr_schedule = LRSchedulePerBatch()
        elif mode == "epoch_coarse":
            epochs = 9
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 10**(epoch - epochs))

        model.compile(loss=loss, optimizer=optimizer)
        history = model.fit(train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[lr_schedule, CustomCallback()])
        TrainingUtils.visualize_lr_selection(history.history.get("lr"), history.history.get("loss"))
        return history

    @staticmethod
    def visualize_lr_selection(lr_values, loss_values, out_dir="."):
        fig, ax = plt.gcf(), plt.gca()
        ax.semilogx(lr_values, loss_values, marker="d")
        loss_range = [max(min(loss_values) - .5, 0), 3 * min(loss_values)]
        axis_range = [1e-8, 1e-1, *loss_range]
        ax.axis(axis_range)
        ax.grid(axis='x')
        time_now = datetime.now().strftime("%H:%M:%S")
        plot_save_path = os.path.join(out_dir, f"lr_search_{time_now}.png")
        fig.savefig(plot_save_path, dpi=120)

    @staticmethod
    def save_model(model: tf.keras.Model):
        time_now = datetime.now().strftime("%H_%M_%S")
        save_path = f"model_{time_now}.h5"
        model.save(save_path, include_optimizer=False, save_format='h5')

    @staticmethod
    def load_model(model_path: str):
        model = tf.keras.models.load_model(model_path)  # type: tf.keras.Model
        return model


def run_image_classification():
    # check_environment()

    # parameters
    VAL_SPLIT = .50
    DATA_DIR = _DATA_DIR
    BS = 64
    IMG_SIZE = (128, 128)

    # get model
    model = get_sample_image_model(input_shape=(128, 128, 3), num_classes=5)
    model.summary()

    # get data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=.1,
                                                                    width_shift_range=0.1,
                                                                    height_shift_range=0.1,
                                                                    zoom_range=0.1,
                                                                    horizontal_flip=True,
                                                                    validation_split=VAL_SPLIT)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=VAL_SPLIT)
    # class_mode: One of "categorical", "binary", "sparse"
    train = train_datagen.flow_from_directory(directory=DATA_DIR, target_size=IMG_SIZE, subset='training',
                                              batch_size=BS, interpolation='bicubic', class_mode='categorical', seed=1)
    validation = validation_datagen.flow_from_directory(directory=DATA_DIR, target_size=IMG_SIZE, subset='validation',
                                                        batch_size=BS, interpolation='bicubic', class_mode='categorical', seed=1)

    # training
    tu = TrainingUtils()
    hist = tu.select_lr(model, train, mode='epoch_coarse')
    # hist = tu.train_model(model, train, validation, lr=.001, loss='categorical_crossentropy', epochs=3)

    # save
    tu.save_model(model)
    tu.save_history(hist)


if __name__ == '__main__':
    run_image_classification()
