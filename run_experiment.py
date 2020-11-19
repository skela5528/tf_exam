import os
import pathlib
import numpy as np
import logging
from datetime import datetime
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import data
from net_bloks import *

_DATA_DIR = "/home/cortica/.keras/datasets/flower_photos"
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s')
LOG = logging.getLogger("tf_exam")


def check_environment():
    print(f"* check packages versions *")
    print(f" - tf version {tf.__version__}")
    print(f" - tfds version {tfds.__version__}\n")


def download_data():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    return data_dir


def get_model_image_classification(input_shape=(256, 256, 3), kernels=(16, 32, 64, 32, 64), classifier_units=(256, 5)):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))
    for bid, k in enumerate(kernels):
        model.add(get_conv_pool_block(k, add_batch_norm=True, bid=bid))
    model.add(tf.keras.layers.GlobalAveragePooling2D())
    model.add(get_classifier(classifier_units, output_activation="softmax"))
    model.summary()
    return model


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


class ExperimentRunner:
    def __init__(self, model: tf.keras.Sequential, train_data, validation_data, out_dir="out"):
        self.model = model
        self.train_data = train_data
        self.validation_data = validation_data
        self.history = None
        time_now = 0  # datetime.now().strftime("%H:%M:%S")
        self.out_dir = f"{out_dir}_{time_now}"
        os.makedirs(self.out_dir, exist_ok=True)

    def select_lr(self, loss='categorical_crossentropy', optimizer=None, epochs=150, batch_size=32, mode="epoch"):
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

        self.model.compile(loss=loss, optimizer=optimizer)
        history = self.model.fit(self.train_data,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 callbacks=[lr_schedule, CustomCallback()])
        self._visualize_lr_selection(history.history.get("lr"), history.history.get("loss"), loss)

    def _visualize_lr_selection(self, lr_values, loss_values, loss_name, axis_range=[1e-8, 1e-1, 0, 2]):
        fig, ax = plt.gcf(), plt.gca()
        ax.semilogx(lr_values, loss_values, marker="d")
        ax.axis(axis_range)
        ax.grid(axis='x')
        time_now = datetime.now().strftime("%H:%M:%S")
        plot_save_path = os.path.join(self.out_dir, f"lr_search_{time_now}.png")
        fig.savefig(plot_save_path, dpi=120)

    def train_model(self, lr=.001, loss='categorical_crossentropy', epochs=3, verbose=1):
        K.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)

        opt = tf.keras.optimizers.RMSprop(learning_rate=lr)
        self.model.compile(opt, loss=loss, metrics=["acc"])
        self.history = self.model.fit(self.train_data,
                                      validation_data=self.validation_data,
                                      epochs=epochs,
                                      verbose=verbose)

    def visualize_training(self):
        pass


if __name__ == '__main__':
    check_environment()

    m = get_model_image_classification()
    train = data.get_image_dataset_train(directory=_DATA_DIR)
    validation = data.get_image_dataset_validation(directory=_DATA_DIR)

    exp_flowers = ExperimentRunner(model=m, train_data=train, validation_data=validation)
    opt = tf.keras.optimizers.RMSprop()
    exp_flowers.select_lr(epochs=3, mode="epoch_coarse", optimizer=opt, loss="sparse_categorical_crossentropy")

    # exp_flowers.train_model(loss='sparse_categorical_crossentropy', epochs=5)
    # print(hist)


