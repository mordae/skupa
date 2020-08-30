#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np

from os.path import join, dirname

from skupa.lms.tracker import Tracker
from skupa.pipe import Worker
from skupa.util import defer


__all__ = ['LandmarkDetector']


MODEL_PATH = join(dirname(__file__), '..', 'model', 'lms-dlib', '68.dat')

MODEL_WIDTH  = 640
MODEL_HEIGHT = 480


class LandmarkDetector(Worker):
    requires = ['frame', 'face']
    provides = ['lms']

    def __init__(self, tracking):
        self.tracking = tracking

    def prepare(self, meta):
        self.meta = meta

        meta['width']  = max(meta.get('width',  0), MODEL_WIDTH)
        meta['height'] = max(meta.get('height', 0), MODEL_HEIGHT)

    async def start(self):
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

        shape = await defer(self.predictor, gray, dlib.rectangle(*job.face))
        lms = np.array([[pt.x, pt.y] for pt in shape.parts()])

        if self.tracking:
            job.lms = self.tracker.track(job.frame, lms)

        else:
            if not self.average.any():
                self.average = lms

            self.average = self.average * 0.5 + lms * 0.5
            job.lms = self.average[:]


# vim:set sw=4 ts=4 et:
