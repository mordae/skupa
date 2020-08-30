#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np

from skupa.pipe import Worker
from skupa.util import defer

MODEL_WIDTH = 640
MODEL_HEIGHT = 480


__all__ = ['FaceDetector']


def rect2arr(rect, **kw):
    return np.array([rect.left(),  rect.top(), rect.right(), rect.bottom()], **kw)


class FaceDetector(Worker):
    requires = ['frame']
    provides = ['face']

    def prepare(self, meta):
        self.meta = meta

        meta['width']  = max(meta.get('width',  0), MODEL_WIDTH)
        meta['height'] = max(meta.get('height', 0), MODEL_HEIGHT)

    async def start(self):
        self.detector = dlib.get_frontal_face_detector()

    async def process(self, job):
        # DLib operates on grayscale images.
        gray = cv2.cvtColor(job.frame, cv2.COLOR_BGR2GRAY)

        # Using 1/2 resolution is way faster and good enough for us,
        # because the tracked person is sitting close to the camera.
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

        # Try to find some faces.
        faces = await defer(self.detector, half)

        # Convert face bounding box to numpy array while scaling it back
        # to the original size.
        faces = [rect2arr(face, dtype=np.int32) * 2 for face in faces]

        # Order faces from left.
        faces = list(sorted(faces, key=lambda r: r[0]))

        # Select the first one.
        # TODO: Maybe choose the middle one?

        if faces:
            job.face = faces[0]
        else:
            job.face = None


# vim:set sw=4 ts=4 et:
