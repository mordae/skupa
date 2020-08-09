#!/usr/bin/env python
# -*- coding: utf-8 -*-

import skupa.lms_dlib
import skupa.lms_onnx


detectors = dict(
    dlib = skupa.lms_dlib.LandmarkDetector,
    onnx = skupa.lms_onnx.LandmarkDetector,
)


# vim:set sw=4 ts=4 et:
