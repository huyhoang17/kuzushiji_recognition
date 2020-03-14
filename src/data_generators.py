import cv2
import numpy as np
import tensorflow
from tensorflow.keras.utils import to_categorical

from utils import (
    norm_mean_std,
    resize_padding,
    load_image,
    load_mask
)


class KuzuDataGenerator(tensorflow.keras.utils.Sequence):
    """Data generator for segmentation model
    """

    def __init__(self,
                 img_fps,
                 labels,
                 batch_size=1,
                 img_size=(512, 512),
                 no_channels=3,
                 n_classes=2,
                 mask_thres=0.5,
                 augment=None,
                 shuffle=True,
                 debug=False):

        self.img_size = img_size
        self.no_channels = no_channels
        self.batch_size = batch_size
        print(">>> Batch_size: {} images".format(self.batch_size))

        self.img_fps = img_fps
        self.labels = labels
        assert len(self.img_fps) == len(self.labels)
        self.ids = range(len(self.img_fps))

        self.n_classes = n_classes
        self.mask_thres = mask_thres
        self.augment = augment
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        X, y = self.__data_generation(temp_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):

        X = np.empty((0, *self.img_size, self.no_channels))  # noqa
        y = np.empty((0, *self.img_size, self.n_classes))  # noqa

        for index, id_ in enumerate(ids):

            img = cv2.imread(self.img_fps[id_])[:, :, ::-1]  # BGR2RGB
            mask = load_mask(img, self.labels[id_])
            mask = cv2.resize(mask, self.img_size)

            # tuning mask
            mask[mask >= self.mask_thres] = 1
            mask[mask < self.mask_thres] = 0

            img = load_image(img, img_size=self.img_size, expand=False)

            if img is None or mask is None:
                continue

            # add augmentation
            if self.augment is not None:
                aug = self.augment(image=img, mask=mask)
                img = aug["image"]
                mask = aug["mask"]

            img = img.astype(np.float32)
            mask = mask.astype(np.float32)

            X = np.vstack((X, np.expand_dims(img, axis=0)))
            y = np.vstack((y, np.expand_dims(mask, axis=0)))

        X = X.astype(np.float32)
        y = y.astype(np.float32)

        assert X.shape[0] == y.shape[0]
        return X, y


class KuzuCharClassifyGenerator(tensorflow.keras.utils.Sequence):
    """Data generator for classification model
    """

    def __init__(self,
                 img_fps,
                 labels,
                 batch_size=64,
                 img_size=(64, 64),
                 no_channels=3,
                 no_classes=3422,
                 augment=None,
                 pad=True,
                 shuffle=True,
                 debug=False):

        self.img_size = img_size
        self.no_channels = no_channels
        self.batch_size = batch_size
        print(">>> Batch_size: {} images".format(self.batch_size))

        self.img_fps = img_fps
        self.labels = labels
        assert len(self.img_fps) == len(self.labels)
        self.ids = range(len(self.img_fps))

        self.no_classes = no_classes
        self.augment = augment
        self.pad = pad
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        temp_ids = [self.ids[k] for k in indexes]
        X, y = self.__data_generation(temp_ids)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, ids):
        X = np.empty((0, *self.img_size, self.no_channels))
        y = []

        for index, id_ in enumerate(ids):
            img = cv2.imread(self.img_fps[id_])[:, :, ::-1]
            label = self.labels[id_]

            if img is None:
                continue
            img = norm_mean_std(img)

            if self.pad:
                img = resize_padding(img, desired_size=self.img_size[0])
            else:
                img = cv2.resize(img, self.img_size)

            if img is None:
                continue

            X = np.vstack((X, np.expand_dims(img, axis=0)))
            y.append(label)

        y = to_categorical(y, num_classes=self.no_classes)
        return X, y
