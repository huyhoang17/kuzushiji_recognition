import os
import time

from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks

from augs import AUGMENT_TRAIN
from consts import (
    MODEL_FD,
    CPS_FD,
    LOG_FD,
    IMG_SIZE_SEGMENT,
    BATCH_SIZE_SEGMENT,
    LR_SEGMENT
)
from data_generators import KuzuDataGenerator
from metrics import (
    my_iou_metric,
    dice_coef
)
from model_segment import resnet_unet
from losses import bce_dice_loss


def train(img_fps, labels):
    net = resnet_unet(
        img_size=(IMG_SIZE_SEGMENT, IMG_SIZE_SEGMENT),
        no_channels=3,
        start_neurons=16
    )
    net.summary()
    print(net.count_params(), net.inputs, net.outputs)

    # train / test split
    train_img_fps, val_img_fps, train_labels, val_labels = \
        train_test_split(img_fps, labels, test_size=0.1, random_state=42)

    # data generator
    train_generator = KuzuDataGenerator(
        train_img_fps, train_labels,
        batch_size=BATCH_SIZE_SEGMENT,
        img_size=(IMG_SIZE_SEGMENT, IMG_SIZE_SEGMENT),
        shuffle=True,
        augment=AUGMENT_TRAIN,
    )
    val_generator = KuzuDataGenerator(
        val_img_fps, val_labels,
        batch_size=BATCH_SIZE_SEGMENT,
        img_size=(IMG_SIZE_SEGMENT, IMG_SIZE_SEGMENT),
        shuffle=False,
        augment=None
    )
    print(len(train_generator), len(val_generator))

    # callbacks
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(CPS_FD, "kuzu_{}_{}_cps.h5".format(
            IMG_SIZE_SEGMENT, time.time()
        )),
        monitor='val_loss', verbose=1, save_best_only=True, mode='min'
    )
    early = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=4, verbose=1)
    redonplat = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.1, mode="min", patience=3, verbose=1
    )
    csv_logger = callbacks.CSVLogger(
        os.path.join(LOG_FD, 'kuzu_log_{}_{}.csv'.format(
            IMG_SIZE_SEGMENT, time.time()
        )),
        append=False, separator=','
    )

    callbacks_list = [
        checkpoint,
        early,
        redonplat,
        csv_logger,
    ]

    # compile
    optim = optimizers.Adam(lr=LR_SEGMENT)
    net.compile(loss=bce_dice_loss, optimizer=optim,
                metrics=[my_iou_metric, dice_coef])

    # fit model
    history = net.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=25,
        callbacks=callbacks_list,
        validation_data=val_generator,
        validation_steps=len(val_generator),
    )

    # save model
    net.save_weights(
        os.path.join(
            MODEL_FD,
            "kuzu_weight_{}_{}.h5".format(IMG_SIZE_SEGMENT, time.time())
        )
    )

    with open(os.path.join(MODEL_FD, "kuzu_config.json"), "w") as f:
        f.write(net.to_json())

    return history


if __name__ == '__main__':
    img_fps, labels = None, None
    train(img_fps, labels)
