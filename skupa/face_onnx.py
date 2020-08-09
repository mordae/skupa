#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Originally taken from:
#   https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch/blob/master/face_onnx/detector.py
#
# License: Apache License 2.0
#

from os.path import join, dirname
from skupa.util import defer

import cv2
import numpy as np
import onnxruntime as ort


__all__ = ['FaceDetector']


MAIN_MODEL_PATH = join(dirname(__file__), 'model',
                       'face-320-linzaer', 'face-320-RFB.onnx')

SLIM_MODEL_PATH = join(dirname(__file__), 'model',
                       'face-320-linzaer', 'face-320-slim.onnx')

MODEL_WIDTH  = 320
MODEL_HEIGHT = 240


def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]


def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])

    return overlap_area / (area0 + area1 - overlap_area + eps)


def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]

    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)

        if 0 < top_k == len(picked) or len(indexes) == 1:
            break

        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(rest_boxes, np.expand_dims(current_box, axis=0))
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


class FaceDetector:
    WIDTH = 640
    HEIGHT = 480

    def __init__(self, model='slim'):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        if model == 'slim':
            model_path = SLIM_MODEL_PATH
        else:
            model_path = MAIN_MODEL_PATH

        self.session = ort.InferenceSession(model_path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name


    def _predict(self, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        picked_labels = []

        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]

            if probs.shape[0] == 0:
                continue

            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.shape[0])

        if not picked_box_probs:
            return np.int32([]), np.array([])

        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= self.WIDTH
        picked_box_probs[:, 1] *= self.HEIGHT
        picked_box_probs[:, 2] *= self.WIDTH
        picked_box_probs[:, 3] *= self.HEIGHT

        return picked_box_probs[:, :4], picked_box_probs[:, 4]


    async def detect(self, orig_image):
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))
        image_mean = np.array([127, 127, 127])
        image = (image - image_mean) / 128
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        image = np.float32(image)

        conf, boxes = await defer(self.session.run, None, {self.input_name: image})
        boxes, probs = self._predict(conf, boxes, 0.8)

        h, w, _ = orig_image.shape

        if len(boxes) > 0:
            boxes[:, 0] *= w / self.WIDTH
            boxes[:, 1] *= h / self.HEIGHT
            boxes[:, 2] *= w / self.WIDTH
            boxes[:, 3] *= h / self.HEIGHT

        # Order faces from left.
        boxes = list(sorted(boxes, key=lambda b: b[0]))

        return np.int32(boxes), probs


# vim:set sw=4 ts=4 et:
