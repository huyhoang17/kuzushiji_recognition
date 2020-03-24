import os
import math
import time
import functools
import random

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from pylab import rcParams
rcParams['figure.figsize'] = 20, 20  # noqa

from utils import (
    make_contours,
    get_centers,
    get_labels,
    vis_pred_bbox,
    filter_polygons_points_intersection,
    vis_pred_bbox_polygon,
    vis_pred_center,
)
from grpc_utils import (
    KuzuSegment,
    KuzuClassify
)

fontsize = 50
font_fp = os.path.abspath('./fonts/NotoSansCJKjp-Regular.otf')
assert os.path.exists(font_fp)
font = ImageFont.truetype(
    font_fp, fontsize, encoding='utf-8'
)


if __name__ == '__main__':
    img_dir = "./images"
    img_fp = os.path.join(img_dir, random.choice(os.listdir(img_dir)))
    print(img_fp)
    filter_polygon = True
    kuzu_seg = KuzuSegment()
    kuzu_cls = KuzuClassify()
    img, origin_image, origin_h, origin_w = kuzu_seg.load_image(img_fp)
    pred_bbox, pred_center = kuzu_seg.predict(img)

    # get all polygon area in image
    polygon_contours = make_contours(pred_bbox)

    # get all center points by contour method
    center_coords = get_centers(pred_center.astype(np.uint8))
    no_center_points = len(center_coords)
    final_center = vis_pred_center(center_coords, rad=2)

    # filter polygon
    if filter_polygon:
        filtered_contours = filter_polygons_points_intersection(polygon_contours, center_coords)  # noqa
        pred_bbox = vis_pred_bbox_polygon(pred_bbox, filtered_contours)
    final_bbox = vis_pred_bbox(pred_bbox, center_coords, width=2)

    y_ratio = origin_h / 512
    x_ratio = origin_w / 512

    pil_img = Image.fromarray(origin_image).convert('RGBA')
    char_canvas = Image.new('RGBA', pil_img.size)
    char_draw = ImageDraw.Draw(char_canvas)

    print(">>> {}".format(no_center_points))
    if no_center_points > 0:
        bbox_cluster = get_labels(center_coords, pred_bbox)

        # ignore background hex color (=0)
        for cluster_index in tqdm(range(len(center_coords))[1:]):

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

            # convert to original coordinates
            x = int(x * x_ratio)
            w = int(w * x_ratio)
            y = int(y * y_ratio)
            h = int(h * y_ratio)

            # set offset to crop character
            offset = 5  # percentage
            y_diff = math.ceil(h * offset / 100)
            x_diff = math.ceil(w * offset / 100)

            # expand area
            y_from = y - y_diff
            y_to = y + h + y_diff
            x_from = x - x_diff
            x_to = x + w + x_diff

            # tune
            y_from, y_to, x_from, x_to = \
                list(map(functools.partial(np.maximum, 0),
                         [y_from, y_to, x_from, x_to]))

            try:
                char_img = origin_image[y_from:y_to, x_from:x_to]
                char_img = kuzu_cls.load_image(char_img)
                pred_label = kuzu_cls.predict(char_img)
                # print(pred_label)

                char_draw.text(
                    (x + w + fontsize / 4, y + h / 2 - fontsize),
                    pred_label, fill=(0, 0, 255, 255),
                    font=font
                )

            except Exception as e:
                print(e)
                continue

    char_img = Image.alpha_composite(pil_img, char_canvas)
    char_img = char_img.convert("RGB")
    char_img = np.asarray(char_img)

    final_bbox = cv2.resize(final_bbox, (origin_w, origin_h))
    final_center = cv2.resize(final_center, (origin_w, origin_h))

    plt.imshow(char_img)
    plt.imshow(final_bbox, cmap="jet", alpha=0.50)
    plt.savefig("./media/{}.jpg".format(time.time()), bbox_inches='tight')
    # imageio.imwrite("../media/foo.jpg", )
