from tensorflow.keras import layers
from tensorflow.keras import models

from resnet import residual_block


def resnet_backbone(input_layer,
                    no_classes=3422,
                    start_neurons=32,
                    dropout_rate=0.1):
    # input_data = layers.Input(name='the_input', shape=IMG_SIZE + (1, ), dtype='float32')  # noqa

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
    input_layer = layers.Input(
        name='input_image', shape=(64, 64, 3), dtype='float32'
    )
    cnet = resnet_backbone(input_layer)
    print(cnet.count_params(), cnet.inputs, cnet.outputs)
