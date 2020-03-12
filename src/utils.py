import copy
from functools import wraps

from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder


with open("./models/le.pkl", "rb") as f:
    le = joblib.load(f)


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
            (x + w + fontsize / 4, y + h / 2 - fontsize),
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
def visual_pred_gt(model, img_fp, bbox_thres=0.01, center_thres=0.02):
    test_id = img_fp.split("/")[-1][:-4]
    img_labels = df_train[df_train["image_id"].isin(
        [test_id])]["labels"].values[0]
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
