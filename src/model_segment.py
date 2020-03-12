from tensorflow.keras import layers

from resnet import residual_block

IMG_SIZE = 512


def resnet_unet(img_size=(512, 512),
                no_channels=3,
                start_neurons=32,
                dropout_rate=0.25):

    # inner
    input_layer = layers.Input(name='the_input',
                               shape=(*img_size, no_channels),  # noqa
                               dtype='float32')

    # down 1
    conv1 = layers.Conv2D(start_neurons * 1, (3, 3),
                          activation=None, padding="same")(input_layer)
    conv1 = residual_block(conv1, start_neurons * 1)
    conv1 = residual_block(conv1, start_neurons * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(dropout_rate)(pool1)

    # down 2
    conv2 = layers.Conv2D(start_neurons * 2, (3, 3),
                          activation=None, padding="same")(pool1)
    conv2 = residual_block(conv2, start_neurons * 2)
    conv2 = residual_block(conv2, start_neurons * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(dropout_rate)(pool2)

    # down 3
    conv3 = layers.Conv2D(start_neurons * 4, (3, 3),
                          activation=None, padding="same")(pool2)
    conv3 = residual_block(conv3, start_neurons * 4)
    conv3 = residual_block(conv3, start_neurons * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(dropout_rate)(pool3)

    # middle
    middle = layers.Conv2D(start_neurons * 8, (3, 3),
                           activation=None, padding="same")(pool3)
    middle = residual_block(middle, start_neurons * 8)
    middle = residual_block(middle, start_neurons * 8, True)

    # up 1
    deconv3 = layers.Conv2DTranspose(
        start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(middle)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(dropout_rate)(uconv3)

    uconv3 = layers.Conv2D(start_neurons * 4, (3, 3),
                           activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 4)
    uconv3 = residual_block(uconv3, start_neurons * 4, True)

    # up 2
    deconv2 = layers.Conv2DTranspose(
        start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv3)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(dropout_rate)(uconv2)

    uconv2 = layers.Conv2D(start_neurons * 2, (3, 3),
                           activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 2)
    uconv2 = residual_block(uconv2, start_neurons * 2, True)

    # up 3
    deconv1 = layers.Conv2DTranspose(
        start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    uconv1 = layers.Dropout(dropout_rate)(uconv1)

    uconv1 = layers.Conv2D(start_neurons * 1, (3, 3),
                           activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 1)
    uconv1 = residual_block(uconv1, start_neurons * 1, True)

    # output mask
    output_layer = layers.Conv2D(
        2, (1, 1), padding="same", activation=None)(uconv1)
    # 2 classes: character mask & center point mask
    output_layer = layers.Activation('sigmoid')(output_layer)

    model = models.Model(inputs=[input_layer], outputs=output_layer)
    return model


if __name__ == '__main__':

    net = resnet_unet(
        img_size=(IMG_SIZE, IMG_SIZE),
        no_channels=3,
        start_neurons=16
    )
    print(net.count_params(), net.inputs, net.outputs)
