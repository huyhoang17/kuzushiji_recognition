import os
import math
import time
import functools
import joblib
import random

from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow_serving.apis import (
    predict_pb2,
    prediction_service_pb2_grpc
)
import grpc

from utils import (
    make_contours,
    get_centers,
    get_labels,
    vis_pred_bbox,
    filter_polygons_points_intersection,
    vis_pred_bbox_polygon,
    vis_pred_center,
    norm_mean_std,
    minmax_scaler
)


class KuzuSegment:

    def __init__(self,
                 img_size=(512, 512),
                 host="localhost",
                 port=8500,
                 input_name="input_image",
                 output_name="pred_mask",
                 model_spec_name="kuzu_segment",
                 model_sig_name="kuzu_segment_sig",
                 timeout=10):

        self.img_size = img_size
        self.input_name = input_name
        self.output_name = output_name

        # init channel
        self.channel = grpc.insecure_channel("{}:{}".format(host, port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )

        # Create PredictRequest ProtoBuf from image data
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_spec_name
        self.request.model_spec.signature_name = model_sig_name
        self.timeout = timeout

    def load_image(self,
                   oimg):

        if isinstance(oimg, str):
            oimg = cv2.imread(oimg)[:, :, ::-1]

        h, w, _ = oimg.shape
        img = cv2.resize(oimg, self.img_size)
        img = norm_mean_std(img)

        img = np.expand_dims(img, axis=0)

        return img, oimg, h, w

    def _grpc_client_request(self, img):

        assert img.ndim == 4
        self.request.inputs[self.input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=np.float32,
                shape=[*img.shape]  # noqa
            )
        )

        # Call the TFServing Predict API
        predict_response = self.stub.Predict(
            self.request, timeout=self.timeout
        )

        return predict_response

    def predict(self,
                img,
                bbox_thres=0.01,
                center_thres=0.02):

        # img = self.load_image(img_fp)
        result = self._grpc_client_request(img)

        # parse result
        pred_mask = tf.contrib.util.make_ndarray(
            result.outputs[self.output_name]
        )
        pred_mask = pred_mask[0]
        pred_bbox, pred_center = pred_mask[:, :, 0], pred_mask[:, :, 1]
        pred_bbox = (pred_bbox > bbox_thres).astype(np.float32)
        pred_center = (pred_center > center_thres).astype(np.float32)

        return pred_bbox, pred_center


class KuzuClassify:

    def __init__(self,
                 img_size=(64, 64),
                 host="localhost",
                 port=8500,
                 input_name="input_image",
                 output_name="y_pred",
                 model_spec_name="kuzu_classify",
                 model_sig_name="kuzu_classify_sig",
                 timeout=10):

        self.img_size = img_size
        self.input_name = input_name
        self.output_name = output_name

        # init channel
        self.channel = grpc.insecure_channel("{}:{}".format(host, port))
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(
            self.channel
        )

        # Create PredictRequest ProtoBuf from image data
        self.request = predict_pb2.PredictRequest()
        self.request.model_spec.name = model_spec_name
        self.request.model_spec.signature_name = model_sig_name
        self.timeout = timeout

        self.load_le()

    def load_le(self):

        le_fp = os.path.abspath("./models/le.pkl")
        assert os.path.exists(le_fp)
        with open(le_fp, "rb") as f:
            self.le = joblib.load(f)

    def deunicode(self,
                  codepoint):
        return chr(int(codepoint[2:], 16))

    def load_image(self,
                   char_img):

        if isinstance(char_img, str):
            char_img = cv2.imread(char_img)[:, :, ::-1]

        char_img = norm_mean_std(char_img)
        char_img = cv2.resize(char_img, (64, 64))
        char_img = np.expand_dims(char_img, axis=0)

        return char_img

    def _grpc_client_request(self, img):

        assert img.ndim == 4
        self.request.inputs[self.input_name].CopyFrom(
            tf.contrib.util.make_tensor_proto(
                img,
                dtype=np.float32,
                shape=[*img.shape]  # noqa
            )
        )

        # Call the TFServing Predict API
        predict_response = self.stub.Predict(
            self.request, timeout=self.timeout
        )

        return predict_response

    def predict(self,
                img):

        # img = self.load_image(img_fp)
        result = self._grpc_client_request(img)

        # parse result
        y_pred = tf.contrib.util.make_ndarray(result.outputs[self.output_name])
        y_pred = y_pred[0]

        y_argmax = np.argmax(y_pred)
        pred_label_unicode = self.le.classes_[y_argmax]
        pred_label = self.deunicode(pred_label_unicode)

        return pred_label
