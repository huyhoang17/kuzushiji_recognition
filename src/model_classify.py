from tensorflow.keras import layers
from tensorflow.keras import models

from consts import (
    IMG_SIZE_CLASSIFY,
    NO_CLASSES
)
from resnet import residual_block


def resnet_backbone(no_classes=3422,
                    no_channels=3,
                    start_neurons=32,
                    dropout_rate=0.1):
    input_layer = layers.Input(
        name='input_image',
        shape=(IMG_SIZE_CLASSIFY, IMG_SIZE_CLASSIFY, no_channels),
        dtype='float32'
    )

    for index, i in enumerate([1, 2, 2, 4, 8]):
        if index == 0:
            inner = input_layer
        inner = layers.Conv2D(start_neurons * i, (3, 3),
                              activation=None, padding="same")(inner)
        inner = residual_block(inner, start_neurons * i)
        inner = residual_block(inner, start_neurons * i, True)

        if i <= 4:
            inner = layers.MaxPooling2D((2, 2))(inner)

        if dropout_rate:
            inner = layers.Dropout(dropout_rate)(inner)

    print(inner.get_shape())
    inner = layers.Flatten()(inner)
    inner = layers.Dense(no_classes, activation="softmax")(inner)
    net = models.Model(inputs=[input_layer], outputs=inner)
    return net


if __name__ == '__main__':

    cnet = resnet_backbone(
        no_classes=NO_CLASSES
    )
    print(cnet.count_params(), cnet.inputs, cnet.outputs)
