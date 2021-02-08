#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import dlib
import numpy as np

from skupa.pipe import Worker
from skupa.util import defer, resize_and_pad

# Using 1/2 resolution is way faster and good enough for us,
# because the tracked person is sitting close to the camera.
MODEL_WIDTH  = 640 // 2
MODEL_HEIGHT = 480 // 2


__all__ = ['FaceDetector']


def rect2arr(rect, **kw):
    return np.array([rect.left(),  rect.top(), rect.right(), rect.bottom()], **kw)


class FaceDetector(Worker):
    requires = ['frame']
    provides = ['face']

    async def prepare(self):
        self.detector = dlib.get_frontal_face_detector()

    async def process(self, job):
        if job.frame is None:
            job.face = None
            return

        # DLib operates on grayscale images.
        gray = cv2.cvtColor(job.frame, cv2.COLOR_BGR2GRAY)

        # Resize to fit the model.
        small, ratio = resize_and_pad(gray, (MODEL_HEIGHT, MODEL_WIDTH))

        # Try to find some faces.
        faces = await defer(self.detector, small)

        # Convert face bounding box to numpy array while scaling it back
        # to the original size.
        faces = [rect2arr(face, dtype=np.int32) for face in faces]

        if faces:
            # Order faces from left.
            faces = list(sorted(faces, key=lambda r: r[0]))

            # Select the left-most one.
            job.face = faces[0]

            # Restore the original ratio.
            job.face = np.round(job.face / ratio).astype(np.int)
        else:
            job.face = None


# vim:set sw=4 ts=4 et:
