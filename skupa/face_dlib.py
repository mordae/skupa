#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, dirname
from skupa.util import defer

import cv2
import dlib
import numpy as np


__all__ = ['FaceDetector']


def rect2arr(rect, **kw):
    return np.array([rect.left(),  rect.top(), rect.right(), rect.bottom()], **kw)


class FaceDetector:
    WIDTH = 640
    HEIGHT = 480

    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()


    async def detect(self, image):
        # DLib operates on grayscale images.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Using 1/2 resolution is way faster and good enough for us,
        # because the tracked person is sitting close to the camera.
        half = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2))

        # Try to find some faces.
        faces = await defer(self.detector, half)
        if not faces:
            return np.int32([]), np.float32([])

        # Convert face bounding box to numpy array while scaling it back
        # to the original size.
        faces = [rect2arr(face, dtype=np.int32) * 2 for face in faces]

        # Order faces from left.
        faces = list(sorted(faces, key=lambda r: r[0]))

        return faces, np.float32([1.0 for _ in faces])


# vim:set sw=4 ts=4 et:
