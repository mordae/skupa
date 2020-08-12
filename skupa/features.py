#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from scipy.spatial.distance import euclidean as distance


__all__ = ['FeatureTracker', 'Features']


class FeatureTracker:
    def __init__(self, index):
        self.index = index

        # Averages for smoothing.
        self.avg_expr = np.float32([0, 0, 0, 0, 0, 0])
        self.avg_rpy  = np.float32([0, 0, 0])

        # Extreme open and closed eye measurements
        self.eyemin = np.float32([99, 99])
        self.eyemax = np.float32([00, 00])

        # Extreme mouth width and height measurements
        self.mthmin = np.float32([99, 99])
        self.mthmax = np.float32([00, 00])


    def track(self, lms, rpy, expr):
        self.avg_expr = expr = (self.avg_expr + expr) / 2.
        self.avg_rpy  = rpy  = (self.avg_rpy  + rpy)  / 2.

        eyes = self._eyes(lms)
        mouth = self._mouth(lms)

        # Allow only one expression at a time.
        largest = max(expr)

        if largest <= 0.4:
            largest = 0.0

        for i, e in enumerate(expr):
            if e != largest:
                expr[i] = 0.0

        # Prevent blinking and mouth shaping while holding an expression.
        if largest > 0.4:
            eyes[:] = 1.0
            mouth[:] = (1.0, 0.0)

        # Return the feature set.
        return Features(self.index, lms, rpy, expr, eyes, mouth)


    def _eyes(self, lms):
        rv1 = distance(lms[37], lms[41])
        rv2 = distance(lms[38], lms[40])

        lv1 = distance(lms[43], lms[47])
        lv2 = distance(lms[44], lms[46])

        # Raw eye heights
        eyes = np.float32([(rv1 + rv2) / 2, (lv1 + lv2) / 2])

        # Slowly reset eye extremes
        self.eyemin += 0.0003
        self.eyemax -= 0.01

        # Output coefficients
        out = np.float32([1.0, 1.0])

        for i in range(2):
            if eyes[i] < self.eyemin[i]:
                self.eyemin[i] = max(0, eyes[i])

            if eyes[i] > self.eyemax[i]:
                self.eyemax[i] = max(0, eyes[i])

            r = [self.eyemin[i], self.eyemax[i]]
            out[i] = np.interp(eyes[i], r, [0.0, 1.0])

        return out


    def _mouth(self, lms):
        # Inside lips vertical distance
        mv1 = distance(lms[61], lms[67])
        mv2 = distance(lms[62], lms[66])
        mv3 = distance(lms[63], lms[65])

        # Horizontal mouth distances from the corner to the middle
        mh1 = distance(lms[60], (lms[57] + lms[51]) / 2)
        mh2 = distance(lms[64], (lms[57] + lms[51]) / 2)

        # Raw mouth dimensions
        mth = np.float32([(mh1 + mh2) / 2, (mv1 + mv2 + mv3) / 3])

        # Slowly reset mouth extremes
        self.mthmin += 0.0001
        self.mthmax -= 0.001

        # Output mouth coefficients
        out = np.float32([1.0, 1.0])

        for i in range(2):
            if mth[i] < self.mthmin[i]:
                self.mthmin[i] = max(0, mth[i])

            if mth[i] > self.mthmax[i]:
                self.mthmax[i] = max(0, mth[i])

            r = [self.mthmin[i], self.mthmax[i]]
            out[i] = np.interp(mth[i], r, [0.0, 1.0])

        return out


class Features:
    def __init__(self, index, lms, rpy, expr, eyes, mouth):
        self.index = index
        self.lms   = lms
        self.rpy   = rpy
        self.expr  = expr
        self.eyes  = eyes
        self.mouth = mouth


# vim:set sw=4 ts=4 et:
