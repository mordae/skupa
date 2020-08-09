#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np

from os.path import join, dirname

from skupa.tracker import Tracker
from skupa.util import defer


MODEL_PATH = join(dirname(__file__), 'model', 'lms-dlib', '68.dat')


class LandmarkDetector:
    WIDTH  = 640
    HEIGHT = 480

    def __init__(self):
        self.predictor = dlib.shape_predictor(MODEL_PATH)
        self.tracker = Tracker()

    async def detect(self, image, box):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        shape = await defer(self.predictor, image, dlib.rectangle(*box))
        lms = np.float32([[pt.x, pt.y] for pt in shape.parts()])

        try:
            return self.tracker.track(image, lms)
        except:
            return lms


# vim:set sw=4 ts=4 et:
