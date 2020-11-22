import tensorflow as tf


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


def get_sample_image_model(input_shape, num_classes, bn=False, dropout_p=0):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=input_shape))

    # block Conv - Pool - Bn
    model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    if bn:
        model.add(tf.keras.layers.BatchNormalization())

    # block Conv - Pool - Bn
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D(pool_size=2))
    if bn:
        model.add(tf.keras.layers.BatchNormalization())

    # Global Pool - Classifier
    model.add(tf.keras.layers.GlobalAveragePooling2D())

    model.add(tf.keras.layers.Dense(128, activation='relu'))
    if dropout_p > 0.0:
        model.add(tf.keras.layers.Dropout(dropout_p))
    out_units = 1
    out_activation = 'sigmoid'
    if num_classes > 2:
        out_units = num_classes
        out_activation = 'softmax'
    model.add(tf.keras.layers.Dense(units=out_units, activation=out_activation))
    return model


def get_sample_text_model():
    pass


def get_sample_time_series_model():
    pass
