#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np

from os.path import join, dirname

from skupa.lms.tracker import Tracker
from skupa.pipe import Worker
from skupa.util import defer, resize_and_pad


__all__ = ['LandmarkDetector']


MODEL_PATH = join(dirname(__file__), '..', 'model', 'lms-dlib', '68.dat')

MODEL_WIDTH  = 320
MODEL_HEIGHT = 240


class LandmarkDetector(Worker):
    requires = ['frame', 'face']
    provides = ['lms']

    def __init__(self, tracking):
        self.tracking = tracking

    async def prepare(self):
        self.predictor = dlib.shape_predictor(MODEL_PATH)
        if self.tracking:
            self.tracker = Tracker()
        else:
            self.average = np.zeros(68)

    async def process(self, job):
        if job.face is None:
            job.lms = None
            return

        gray = cv2.cvtColor(job.frame, cv2.COLOR_BGR2GRAY)
        small, ratio = resize_and_pad(gray, (MODEL_HEIGHT, MODEL_WIDTH))

        face = (job.face * ratio).astype(np.int)
        shape = await defer(self.predictor, small, dlib.rectangle(*face))
        lms = np.array([[pt.x, pt.y] for pt in shape.parts()]) / ratio

        if self.tracking:
            try:
                job.lms = self.tracker.track(job.frame, lms)
            except KeyboardInterrupt:
                raise
            except:
                print('Tracker failed...')
                job.lms = lms

        else:
            if not self.average.any():
                self.average = lms

            self.average = self.average * 0.5 + lms * 0.5
            job.lms = self.average[:]


# vim:set sw=4 ts=4 et:
