import tensorflow as tf

from tensorflow.keras.applications import vgg16, resnet


def get_conv_pool_block(filters, kernel_size=3, activation='relu', add_batch_norm=False, bid=-1):
    name = f"Conv-{filters}-MaxPool-{bid}"
    block = tf.keras.Sequential(name=name)
    block.add(tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, padding='same', activation=activation))
    if add_batch_norm:
        block.add(tf.keras.layers.BatchNormalization())
    block.add(tf.keras.layers.MaxPool2D())
    return block


def get_classifier(units=(128, 1), output_activation="sigmoid"):
    assert len(units) > 1
    sep = "-"
    name = f"Classifier-{sep.join(map(str, units))}-{output_activation}"
    classifier = tf.keras.Sequential(name=name)
    for i in range(len(units)):
        activation = 'relu' if i < len(units) - 1 else output_activation
        classifier.add(tf.keras.layers.Dense(units=units[i], activation=activation))
    return classifier


def get_simple_text_model():
    pass


def get_simple_time_series_model():
    pass
