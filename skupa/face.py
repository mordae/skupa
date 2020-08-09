#!/usr/bin/env python
# -*- coding: utf-8 -*-

import skupa.face_dlib
import skupa.face_onnx


detectors = dict(
    dlib = skupa.face_dlib.FaceDetector,
    onnx = skupa.face_onnx.FaceDetector,
)


# vim:set sw=4 ts=4 et:
