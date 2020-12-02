import os
import json
import pathlib
import logging
import numpy as np
from typing import Tuple
from datetime import datetime
from matplotlib import pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer

from net_bloks import *


# TODO 1 - prepare sample models for text / for time seq
# TODO 2 - train a model for each topic - take data from 'tfds'
# TODO 3 practice save/ load/ transfer_learning/
# TODO 4 explore tf.data.Dataset interface
# TODO 5 transfer learning
# TODO 5.1 text load pretrained embeddings + text generation

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


def download_fmnist():
    fmnist = tf.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fmnist.load_data()
    return (x_train, y_train), (x_test, y_test)


def download_imdb():
    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=('train[:60%]', 'train[60%:]', 'test'),
        as_supervised=True)
    return train_data, validation_data


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
    def save_model(model: tf.keras.Model):
        time_now = datetime.now().strftime("%H_%M_%S")
        save_path = f"model_{time_now}.h5_"
        model.save(save_path, include_optimizer=False, save_format='h5')

    @staticmethod
    def load_model(model_path: str):
        model = tf.keras.models.load_model(model_path)  # type: tf.keras.Model
        return model

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
    def select_lr(model, train_data, loss='categorical_crossentropy', optimizer=None, epochs=150, batch_size=32,
                  mode="epoch"):
        assert mode in ["epoch", "batch", "epoch_coarse"], LOG.error(f"wrong mode: {mode}!!!")
        initial_lr = 1e-5
        K.clear_session()
        if optimizer is None:
            optimizer = tf.keras.optimizers.SGD(momentum=0.9)
            optimizer._set_hyper("learning_rate", initial_lr)

        lr_schedule = None
        if mode == "epoch":
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: initial_lr * 10 ** (epoch / 20))
        elif mode == "batch":
            lr_schedule = LRSchedulePerBatch()
        elif mode == "epoch_coarse":
            epochs = 6
            lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: float(10 ** (epoch - epochs + 1)))

        model.compile(loss=loss, optimizer=optimizer)
        history = model.fit(train_data,
                            epochs=epochs,
                            batch_size=batch_size,
                            callbacks=[lr_schedule, CustomCallback()])
        TrainingUtils.visualize_lr_selection(history.history.get("lr"), history.history.get("loss"), mode=mode)
        return history

    @staticmethod
    def visualize_lr_selection(lr_values, loss_values, mode, out_dir="."):
        fig, ax = plt.gcf(), plt.gca()
        marker = 'd'
        loss_range = [max(min(loss_values) - .5, 0), np.percentile(loss_values, 80)]
        if mode == 'epoch':
            marker = ''
        ax.semilogx(lr_values, loss_values, lw=1, marker=marker)
        lr_range = (min(lr_values), max(lr_values))
        axis_range = [1e-8, 1e-1, *loss_range]
        ax.axis(axis_range)
        ax.grid(axis='x')
        # save
        time_now = datetime.now().strftime("%H:%M:%S")
        plot_save_path = os.path.join(out_dir, f"plot_lr_{time_now}.png")
        fig.savefig(plot_save_path, dpi=120)

    @staticmethod
    def visualize_training(loss_train, loss_validation, acc_train, acc_validation, out_dir='.'):
        fig, axs = plt.subplots(1, 2)
        fig.set_size_inches(18, 6)
        axs[0].plot(loss_train, marker='.', label='train', lw=1)
        axs[0].plot(loss_validation, marker='.', label='validation', lw=1)
        axs[0].set_xlabel("#epoch")
        axs[0].set_ylabel("Loss")
        axs[0].yaxis.grid(True)
        ylim = (0, 5) if max(loss_validation) > 2 else (0, 1)
        axs[0].set_ylim(*ylim)
        axs[0].legend()

        axs[1].plot(acc_train, marker='.', label='train', lw=1)
        axs[1].plot(acc_validation, marker='.', label='validation', lw=1)
        axs[1].set_xlabel("#epoch")
        axs[1].set_ylabel("ACC")
        axs[1].set_ylim(.5, 1.0)
        axs[1].yaxis.grid(True)
        axs[1].legend()
        # save
        time_now = datetime.now().strftime("%H:%M:%S")
        plot_save_path = os.path.join(out_dir, f"plot_train_{time_now}.png")
        fig.tight_layout()
        fig.savefig(plot_save_path, dpi=120)

    @staticmethod
    def train_model(model: tf.keras.Model, train_data, validation_data, optimizer, loss='categorical_crossentropy',
                    epochs=3, verbose=1, batch_size=None, callbacks=None):
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
                            verbose=verbose,
                            callbacks=callbacks,
                            batch_size=batch_size)
        return history


def show_data(batch_data: np.ndarray):
    pass


def run_image_classification():
    # check_environment()

    # params data #
    DATA_DIR = _DATA_DIR
    VAL_SPLIT = .2
    IMG_SIZE = (28, 28)
    CHANNELS = 1
    N_CLASSES = 10

    # params model #
    BATCH_NORM = True

    # params training #
    LR = 10 ** -1
    BS = 512
    N_EPOCHS = 3

    # get model
    model = get_sample_image_model(input_shape=(*IMG_SIZE, CHANNELS), num_classes=N_CLASSES, bn=BATCH_NORM)
    model.summary()

    # get data
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=.1,
                                                                    width_shift_range=0.1,
                                                                    height_shift_range=0.1,
                                                                    zoom_range=0.1,
                                                                    horizontal_flip=True,
                                                                    validation_split=VAL_SPLIT,
                                                                    rescale=1/255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(validation_split=VAL_SPLIT,
                                                                         rescale=1/255)

    # flowers data
    # # class_mode: One of "categorical", "binary", "sparse"
    # train = train_datagen.flow_from_directory(directory=DATA_DIR, target_size=IMG_SIZE, subset='training',
    #                                           batch_size=BS, interpolation='bicubic', class_mode='categorical', seed=1)
    # validation = validation_datagen.flow_from_directory(directory=DATA_DIR, target_size=IMG_SIZE, subset='validation',
    #                                                     batch_size=BS, interpolation='bicubic', class_mode='categorical', seed=1)

    # fashion_mnist data
    (x_train, labels_train), (x_validation, labels_validation) = download_fmnist()
    x_train = x_train[:, :, :, np.newaxis]
    x_validation = x_validation[:, :, :, np.newaxis]
    train = train_datagen.flow(x_train, labels_train, batch_size=BS)
    validation = validation_datagen.flow(x_validation, labels_validation, batch_size=BS)

    # training
    tu = TrainingUtils()
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath="best_model",
                                                            save_weights_only=True,
                                                            monitor='val_acc',
                                                            save_best_only=True)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda ep, lr: lr if ep < 20 else lr / 2)

    # Select LR #
    # hist = tu.select_lr(model, train, mode='epoch_coarse', loss=tf.keras.losses.sparse_categorical_crossentropy, batch_size=BS)

    # Train #
    hist = tu.train_model(model, train, validation, opt,
                          loss='sparse_categorical_crossentropy',
                          epochs=N_EPOCHS,
                          callbacks=[save_best_callback, lr_callback])

    # Save #
    # tu.save_model(model)
    # tu.save_history(hist)
    # tu.visualize_training(hist.history['loss'], hist.history.get('val_loss', []), hist.history['acc'], hist.history.get('val_acc', []))

    # # LOAD best model
    # model.load_weights("best_model").expect_partial()
    # model.compile(opt, loss="sparse_categorical_crossentropy", metrics=["acc"])
    # model.evaluate(validation)
    # tu.save_model(model)


def prepare_text_data(data_iter, tokenizer=None, num_words=10000, oov_token="<oov>",
                      remove_stop_words=True) -> Tuple[np.array, np.array, Tokenizer]:
    MAX_SEQ_LEN = 120
    PADDING = 'post'
    TRUNCATING = 'post'
    sentences = []
    labels = []

    stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
                 "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
                 "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
                 "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
                 "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's",
                 "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only",
                 "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd",
                 "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs",
                 "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
                 "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
                 "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's",
                 "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll",
                 "you're", "you've", "your", "yours", "yourself", "yourselves"]

    for sentence, label in data_iter:
        sentence = sentence.numpy().decode('utf-8').strip()
        label = int(label.numpy())

        if remove_stop_words:
            for word in stopwords:
                token = " " + word + " "
                sentence = sentence.replace(token, " ")
        sentences.append(sentence)
        labels.append(label)
    print(f'sentences n : {len(sentences)}')

    # Tokenizing - TRAIN
    if tokenizer is None:
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words, oov_token=oov_token)
        tokenizer.fit_on_texts(sentences)

    sequences = tokenizer.texts_to_sequences(sentences)
    sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences,
                                                                     maxlen=MAX_SEQ_LEN,
                                                                     padding=PADDING,
                                                                     truncating=TRUNCATING)

    return sequences_padded, labels, tokenizer


def run_text_exp():
    # params training #
    LR = 3 * 10 ** -4
    BS = 512
    N_EPOCHS = 10

    model = get_sample_text_model(10000, num_classes=2)
    model.summary()

    # imdb data
    train_data, validation_data = download_imdb()

    train_sequences, train_labels, tok = prepare_text_data(list(train_data), num_words=10 ** 4, remove_stop_words=True)
    validation_sequences, validation_labels, _ = prepare_text_data(validation_data, tokenizer=tok, remove_stop_words=True)

    train_gen = tf.data.Dataset.from_tensor_slices((train_sequences.tolist(), train_labels)).batch(BS)
    validation_gen = tf.data.Dataset.from_tensor_slices((validation_sequences.tolist(), validation_labels)).batch(BS)

    # Training #
    tu = TrainingUtils()
    opt = tf.keras.optimizers.Adam()
    # history = tu.select_lr(model, train_gen, tf.keras.losses.binary_crossentropy, opt, mode='epoch', epochs=100)
    # tu.save_history(history, 'lr_hist')

    hist = tu.train_model(model,
                          train_gen,
                          validation_gen,
                          opt,
                          loss='binary_crossentropy', epochs=N_EPOCHS)

    tu.visualize_training(hist.history['loss'], hist.history.get('val_loss', []),
                          hist.history['acc'], hist.history.get('val_acc', []))


if __name__ == '__main__':
    # check_environment()
    # run_image_classification()

    run_text_exp()



