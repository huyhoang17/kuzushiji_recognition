import copy
import time
from functools import wraps

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
from skimage import measure
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import tensorflow as tf

from consts import FONT_SIZE


# font = ImageFont.truetype(
#     '../fonts/NotoSansCJKjp-Regular.otf',
#     FONT_SIZE, encoding='utf-8'
# )
# with open("./models/le2.pkl", "rb") as f:
#     le = joblib.load(f)


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
    """
    e.g: U+770C --> 県
    """
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


def timer(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(">>> Function {} tooks {}'s".format(func.__name__, end - start))
        return result

    return wrapper


def prepare_targets(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    y_train_enc = le.transform(y_train)
    y_test_enc = le.transform(y_test)
    return le, y_train_enc, y_test_enc


@timer
def visualize_training_data(image_fn,
                            labels,
                            width=3,
                            y_first=False):

    # Convert annotation string to array
    labels = np.array(labels.split(' ')).reshape(-1, 5)

    # Read image
    imsource = Image.open(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    # Separate canvases for boxes and chars
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, *args in labels:  # noqa
        if y_first:
            y, x, h, w = args
        else:
            x, y, w, h = args

        x, y, w, h = int(x), int(y), int(w), int(h)
        try:
            # Convert codepoint to actual unicode character
            char = unicode_map[codepoint]
        except KeyError:
            # some codepoint not exists in unicode_map :/
            print(codepoint)
            continue
        # Draw bounding box around character, and unicode character next to it
        bbox_draw.rectangle(
            (x, y, x + w, y + h), fill=(255, 255, 255, 0),
            outline=(255, 0, 0, 255), width=width
        )
        char_draw.text(
            (x + w + FONT_SIZE / 4, y + h / 2 - FONT_SIZE),
            char, fill=(0, 0, 255, 255),
            font=font
        )

    imsource = Image.alpha_composite(
        Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    # Remove alpha for saving in jpg format.
    imsource = imsource.convert("RGB")
    return np.asarray(imsource)


# utils
def get_centers(mask):
    """find center points by using contour method

    :return: [(y1, x1), (y2, x2), ...]
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
        else:
            cx, cy = cnt[0][0]
        cy = int(np.round(cy))
        cx = int(np.round(cx))
        centers.append([cy, cx])
    centers = np.array(centers)
    return centers


def get_labels(center_coords,
               pred_bbox):
    kmeans = KMeans(len(center_coords), init=center_coords)
    kmeans.fit(center_coords)  # noqa

    x, y = np.where(pred_bbox > 0)
    pred_cluster = kmeans.predict(list(zip(x, y)))

    pred_bbox_ = copy.deepcopy(pred_bbox)
    pred_bbox_[x, y] = pred_cluster

    return pred_bbox_


def draw_rects(center_coords,
               bbox_cluster,
               o_img):
    img = copy.deepcopy(o_img)
    for cluster_index in range(len(center_coords))[1:]:
        char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

        horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
        vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
        x_min, x_max = horizontal_indicies[[0, -1]]
        y_min, y_max = vertical_indicies[[0, -1]]

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    return img


@timer
def get_prediction(model,
                   img_fp,
                   bbox_thres=0.01,
                   center_thres=0.02,
                   show=True):
    print(img_fp)
    o_img = load_image(img_fp, expand=False)
    o_img = np.expand_dims(o_img, axis=0)

    # predict
    start = time.time()
    pred_mask = model.predict(o_img)
    print(">>> Inference time: {}'s".format(time.time() - start))
    pred_bbox, pred_center = pred_mask[0][:, :, 0], pred_mask[0][:, :, 1]
    pred_bbox = (pred_bbox > bbox_thres).astype(np.float32)
    pred_center = (pred_center > center_thres).astype(np.float32)
    assert pred_bbox.shape == pred_center.shape

    center_coords = get_centers(pred_center.astype(np.uint8))
    no_center_points = len(center_coords)
    print(">>> N.o center points: {}".format(no_center_points))
    if len(center_coords) == 0:
        print(">>> Non-text")
        plt.imshow(np.squeeze(o_img))
        return
    bbox_cluster = get_labels(center_coords, pred_bbox)

    plt_img = draw_rects(center_coords, bbox_cluster, np.squeeze(o_img))
    return center_coords, o_img[0], plt_img, pred_bbox, pred_center, plt_img, bbox_cluster  # noqa


@timer
def visual_pred_gt(model,
                   img_fp,
                   img_labels,
                   bbox_thres=0.01,
                   center_thres=0.02):

    test_id = img_fp.split("/")[-1][:-4]
    # img_labels = df_train[df_train["image_id"].isin(
    #     [test_id])]["labels"].values[0]
    char_labels = np.array(img_labels.split(' ')).reshape(-1, 5)

    # visual gt
    img = visualize_training_data(img_fp, img_labels, width=5)

    # visual pred
    oimg, oh, ow = load_image(img_fp, return_hw=True)
    oimg = np.expand_dims(oimg, axis=0)

    start = time.time()
    pred_mask = model.predict(oimg)
    print(">>> Inference time: {}'s".format(time.time() - start))
    pred_bbox, pred_center = pred_mask[0][:, :, 0], pred_mask[0][:, :, 1]
    pred_bbox = (pred_bbox > bbox_thres).astype(np.float32)
    pred_center = (pred_center > center_thres).astype(np.float32)
    assert pred_bbox.shape == pred_center.shape

    center_coords = get_centers(pred_center.astype(np.uint8))
    no_center_points = len(center_coords)

    print(">>> no predicted center: {}".format(no_center_points))
    print(">>> Gt no center points: {}".format(len(char_labels)))
    if len(center_coords) == 0:
        print(">>> Non-text")
        return img

    y_ratio = oh / 512
    x_ratio = ow / 512
    print(y_ratio, x_ratio)

    # draw centers
    print(center_coords.shape)
    for y, x in center_coords:
        x = int(x * x_ratio)
        y = int(y * y_ratio)
        cv2.circle(img, (x, y), 3, (0, 255, 0), 5)

    if no_center_points > 0:
        bbox_cluster = get_labels(center_coords, pred_bbox)

        for cluster_index in range(len(center_coords))[1:]:
            char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

            try:
                horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
                vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
                x_min, x_max = horizontal_indicies[[0, -1]]
                y_min, y_max = vertical_indicies[[0, -1]]
            except IndexError:
                continue

            x = x_min
            y = y_min
            w = x_max - x_min
            h = y_max - y_min

            # resize to origin yx
            x = int(x * x_ratio)
            w = int(w * x_ratio)
            y = int(y * y_ratio)
            h = int(h * y_ratio)

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 3)

    return img


def make_contours(masks, flatten=True):
    """
    flatten: follow by coco's api
    """
    if masks.ndim == 2:
        masks = np.expand_dims(masks, axis=-1)

    masks = masks.transpose((2, 0, 1))

    segment_objs = []
    for mask in masks:
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            contour = np.flip(contour, axis=1)
            if flatten:
                segmentation = contour.ravel().tolist()
            else:
                segmentation = contour.tolist()
            segment_objs.append(segmentation)

    return segment_objs


def filter_polygons_points_intersection(polygon_contours, center_coords):
    """https://github.com/huyhoang17/machine-learning-snippets/blob/master/filter_polygons_points_intersection.py
    """
    # checking if polygon contains point
    final_cons = []
    for con in polygon_contours:
        polygon = Polygon(zip(con[::2], con[1::2]))
        for center in center_coords:
            point = Point(center[1], center[0])
            if polygon.contains(point):
                final_cons.append(con)
                break

    return final_cons


def vis_pred_bbox_polygon(pred_bbox, cons):
    """
    pred_bbox: 1st mask
    cons: list contours return from `make_contours` method
    """
    mask_ = Image.new('1', (512, 512))
    mask_draw = ImageDraw.ImageDraw(mask_, '1')

    for contour in cons:
        mask_draw.polygon(contour, fill=1)

    mask_ = np.array(mask_).astype(np.uint8)
    return mask_ * 255


def vis_pred_center(center_points, rad=2, img_size=(512, 512)):

    # center_points = get_centers(pred_center.astype(np.uint8))

    img = np.zeros((512, 512))
    pil_img = Image.fromarray(img).convert('RGBA')
    center_canvas = Image.new('RGBA', pil_img.size)
    center_draw = ImageDraw.Draw(center_canvas)

    for point in center_points:
        y, x = point
        # x1, y1, x2, y2
        center_draw.ellipse(
            (x - rad, y - rad, x + rad, y + rad), fill='blue', outline='blue'
        )

    res_img = Image.alpha_composite(pil_img, center_canvas)
    res_img = res_img.convert("RGB")
    res_img = np.asarray(res_img)

    return res_img


def vis_pred_bbox(pred_bbox, center_coords, width=6):
    """
    pred_bbox: 1st mask
    center_coords: list of center point coordinates [[x1, y1], [x2, y2], ...]
    """

    bbox_cluster = get_labels(center_coords, pred_bbox)

    img = np.zeros((512, 512))
    pil_img = Image.fromarray(img).convert('RGBA')
    bbox_canvas = Image.new('RGBA', pil_img.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
#     center_canvas = Image.new('RGBA', pil_img.size)
#     center_draw = ImageDraw.Draw(center_canvas)

    # exclude background index
    for cluster_index in range(len(center_coords))[1:]:
        char_pixel = (bbox_cluster == cluster_index).astype(np.float32)

        horizontal_indicies = np.where(np.any(char_pixel, axis=0))[0]
        vertical_indicies = np.where(np.any(char_pixel, axis=1))[0]
        x_min, x_max = horizontal_indicies[[0, -1]]
        y_min, y_max = vertical_indicies[[0, -1]]

        # draw polygon
        bbox_draw.rectangle(
            (x_min, y_min, x_max, y_max), fill=(255, 255, 255, 0),
            outline=(255, 0, 0, 255), width=width
        )
        # draw center

    res_img = Image.alpha_composite(pil_img, bbox_canvas)
    res_img = res_img.convert("RGB")
    res_img = np.asarray(res_img)

    # normalize image
    res_img = res_img / 255
    res_img = res_img.astype(np.float32)

    return res_img
