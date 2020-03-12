from tensorflow.keras import layers


def batch_activate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def convolution_block(x,
                      filters,
                      size,
                      strides=(1, 1),
                      padding='same',
                      activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation:
        x = batch_activate(x)
    return x


def residual_block(block_input,
                   num_filters=16,
                   use_batch_activate=False):
    x = batch_activate(block_input)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, block_input])
    if use_batch_activate:
        x = batch_activate(x)
    return x
