import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from utils import le


def norm_mean_std(img):
    img = img / 255
    img = img.astype('float32')

    mean = np.mean(img, axis=(0, 1, 2))
    std = np.std(img, axis=(0, 1, 2))

    img = (img - mean) / std
    return img


def load_image(img,
               img_size=(512, 512),
               expand=False,
               return_hw=False):

    if isinstance(img, str):
        img = cv2.imread(img)[:, :, ::-1]

    h, w, _ = img.shape
    img = norm_mean_std(img)
    img = cv2.resize(img, img_size)

    if expand:
        img = np.expand_dims(img, axis=0)

    if return_hw:
        return img, h, w
    return img


def get_mask(img, labels):
    """Reference
    """
    mask = np.zeros((img.shape[0], img.shape[1], 2), dtype='float32')
    if isinstance(labels, str):
        labels = np.array(labels.split(' ')).reshape(-1, 5)
        for char, x, y, w, h in labels:
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x + w >= img.shape[1] or y + h >= img.shape[0]:
                continue
            mask[y: y + h, x: x + w, 0] = 1
            radius = 6
            mask[y + h // 2 - radius: y + h // 2 + radius + 1, x +
                 w // 2 - radius: x + w // 2 + radius + 1, 1] = 1
    return mask


def load_mask(img, label):
    try:
        mask = get_mask(img, label)
        mask = mask.astype(np.float32)
    except Exception:
        mask = None

    return mask


def resize_padding(img, desired_size=640):
    """https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """
    ratio = float(desired_size) / max(img.shape)
    new_size = tuple([int(dim * ratio) for dim in img.shape[:2]])

    # resize img
    rimg = cv2.resize(img, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # make padding
    color = [0, 0, 0]
    rimg = cv2.copyMakeBorder(rimg, top, bottom, left,
                              right, cv2.BORDER_CONSTANT, value=color)

    return rimg


def deunicode(char):
    return chr(int(char[2:], 16))


def minmax_scaler(img):
    img = ((img - img.min()) * (1 / (img.max() - img.min()) * 255)).astype('uint8')  # noqa
    return img


def show_arrs(imgs, rows=4, cols=5):
    fig = plt.figure(figsize=(16, 16))
    for i in range(1, cols * rows + 1):
        img = imgs[i - 1]
        fig.add_subplot(rows, cols, i)
        plt.imshow(img)
    plt.show()


def show_imgs(img_fps,
              net=None,
              font=None,
              norm=False,
              pad=False,
              img_size=64,
              rows=4,
              cols=5):
    fig = plt.figure(figsize=(16, 16))
    for i in range(1, cols * rows + 1):
        img = mpimg.imread(img_fps[i - 1])

        # for some unknown reasons its cannot plot on CPU, raise ValueError :D
        # ValueError: Floating point image RGB values must be in the 0..1 range.  # noqa
        if norm:
            img = norm_mean_std(img)

        if pad:
            img = resize_padding(img, img_size)
        else:
            img = cv2.resize(img, (img_size, img_size))

        ax = fig.add_subplot(rows, cols, i)
        # TODO: add font size
        if font is not None:
            ax.title.set_font_properties(font)

        if net is not None:
            pimg = np.expand_dims(img, axis=0)
            y_pred = net.predict(pimg)[0]
            y_argmax = np.argmax(y_pred)
            pred_label_unicode = le.classes_[y_argmax]
            pred_label = deunicode(pred_label_unicode)
            ax.title.set_text(pred_label)

        if not tf.test.is_gpu_available():
            img = minmax_scaler(img)

        plt.imshow(img)

    plt.show()
    return ax



