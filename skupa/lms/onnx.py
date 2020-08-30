#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Originally taken from:
#   https://github.com/ainrichman/Peppa-Facial-Landmark-PyTorch/blob/master/onnx_detector.py
#
# License: Apache License 2.0
#

import cv2
import numpy as np
import onnxruntime as ort

from os.path import join, dirname

from skupa.lms.tracker import Tracker
from skupa.util import defer
from skupa.pipe import Worker


MODEL_PATH = join(dirname(__file__), '..', 'model',
                  'lms-160-ainrichman', 'lms-160-slim.onnx')

MODEL_WIDTH  = 160
MODEL_HEIGHT = 160


class LandmarkDetector(Worker):
    requires = ['frame', 'face']
    provides = ['lms']

    def __init__(self, tracking):
        self.tracking = tracking

    def prepare(self, meta):
        self.meta = meta

        meta['width']  = max(meta.get('width',  0), 320)
        meta['height'] = max(meta.get('height', 0), 240)

    async def start(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        self.session = ort.InferenceSession(MODEL_PATH, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name

        if self.tracking:
            self.tracker = Tracker()
        else:
            self.average = np.zeros(68)

    def _crop_image(self, orig, box):
        box = box.copy()
        image = orig.copy()

        box_width = box[2] - box[0]
        box_height = box[3] - box[1]

        face_width = (1 + 2 * 0.2) * box_width
        face_height = (1 + 2 * 0.2) * box_height

        center = [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]

        box[0] = max(0, center[0] - face_width // 2)
        box[1] = max(0, center[1] - face_height // 2)
        box[2] = min(image.shape[1], center[0] + face_width // 2)
        box[3] = min(image.shape[0], center[1] + face_height // 2)
        box = np.int32(box)

        crop_image = image[box[1]:box[3], box[0]:box[2], :]
        h, w, _ = crop_image.shape
        crop_image = cv2.resize(crop_image, (MODEL_WIDTH, MODEL_HEIGHT))

        return crop_image, ([h, w, box[1], box[0]])

    async def process(self, job):
        if job.face is None or job.frame is None:
            job.lms = None
            return

        crop_image, detail = self._crop_image(job.frame, job.face)
        crop_image = (crop_image - 127.0) / 127.0
        crop_image = np.float32([np.transpose(crop_image, (2, 0, 1))])

        res = await defer(self.session.run, None, {self.input_name: crop_image})
        lms = res[0][0]
        lms = lms[0:136].reshape((-1, 2))
        lms[:, 0] = lms[:, 0] * detail[1] + detail[3]
        lms[:, 1] = lms[:, 1] * detail[0] + detail[2]

        if self.tracking:
            job.lms = self.tracker.track(job.frame, lms)

        else:
            if not self.average.any():
                self.average = lms

            self.average = self.average * 0.5 + lms * 0.5
            job.lms = self.average[:]


# vim:set sw=4 ts=4 et:
