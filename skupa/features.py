#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import numpy as np

from filterpy.gh import GHFilter
from scipy.spatial.distance import euclidean as distance


__all__ = ['FeatureTracker', 'Features']


class FeatureTracker:
    def __init__(self, index):
        self.index = index

        self.gh_mouth = GHFilter(0, 0, 1.0, 0.8, 0.2)
        self.gh_expr  = GHFilter(0, 0, 1.0, 0.6, 0.2)
        self.gh_rpy   = GHFilter(0, 0, 1.0, 0.5, 0.1)

        # Extreme open and closed eye measurements for averages
        self.eyemin = np.float32([2, 2])
        self.eyeavg = np.float32([1, 1])
        self.eyemax = np.float32([0, 0])

        # Open and closed eye history measurements
        self.eyemina = np.float32([[2] * 30] * 2)
        self.eyemaxa = np.float32([[0] * 30] * 2)


    def track(self, lms, rpy, expr):
        eyes = self._eyes(lms)

        # Apply filters to smooth out transitions.
        mouth, _ = self.gh_mouth.update(self._mouth(lms))
        expr, _  = self.gh_expr.update(np.float32(expr))
        rpy, _   = self.gh_rpy.update(rpy)

        # Clip mouth to the 0.0 - 1.0 range.
        mouth[mouth < 0.0] = 0.0
        mouth[mouth > 1.0] = 1.0

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
        # Vertical distance
        rv1 = distance(lms[37], lms[41])
        rv2 = distance(lms[38], lms[40])

        # Horizontal distance
        rh1 = distance(lms[36], lms[39])

        # Ratio
        r_eye = (rv1 + rv2) / rh1 * 2.0

        # Vertical distance
        lv1 = distance(lms[43], lms[47])
        lv2 = distance(lms[44], lms[46])

        # Horizontal distance
        lh1 = distance(lms[42], lms[45])

        # Left eye ratio
        l_eye = (lv1 + lv2) / lh1 * 2.0

        # Raw eye ratios
        eyes = np.float32([r_eye, l_eye])

        # Output eye openness coefficients
        out = np.float32([1.0, 1.0])

        # Update eye limits
        for i in range(2):
            if eyes[i] < self.eyemin[i]:
                self.eyemin[i] = eyes[i]
            elif eyes[i] > self.eyemax[i]:
                self.eyemax[i] = eyes[i]

            self.eyeavg[i] = np.average([self.eyemin[i], self.eyemax[i]])

            if eyes[i] >= self.eyeavg[i]:
                self.eyemaxa[i] = np.roll(self.eyemaxa[i], -1)
                self.eyemaxa[i][-1] = eyes[i]
            else:
                self.eyemina[i] = np.roll(self.eyemina[i], -1)
                self.eyemina[i][-1] = eyes[i]

            eyemina = np.average(self.eyemina[i]) * 1.1
            eyemaxa = np.average(self.eyemaxa[i]) * 0.9

            # Interpolate to the 0.0 - 1.0 range
            out[i] = np.interp(eyes[i], [eyemina, eyemaxa], [0.0, 1.0])

        return out


    def _mouth(self, lms):
        # Nose horizontal distance as a stable reference distance
        nh1 = distance((lms[31] + lms[32]) / 2,
                       (lms[34] + lms[35]) / 2)

        # Inside lips vertical distance
        mv1 = distance(lms[61], lms[67])
        mv2 = distance(lms[62], lms[66])
        mv3 = distance(lms[63], lms[65])

        # Horizontal mouth distances from the corner to the middle
        mh1 = distance(lms[60], (lms[57] + lms[51]) / 2)
        mh2 = distance(lms[64], (lms[57] + lms[51]) / 2)

        # Mouth vertical ratio to nose width
        height = (mv1 + mv2 + mv3) / (nh1 * 3.0) / 1.5

        # Mouth horizontal ratio to nose width
        width = (mh1 + mh2) / nh1 - 2.0

        return np.float32([width, height])


class Features:
    def __init__(self, index, lms, rpy, expr, eyes, mouth):
        self.index = index
        self.lms   = lms
        self.rpy   = rpy
        self.expr  = expr
        self.eyes  = eyes
        self.mouth = mouth


# vim:set sw=4 ts=4 et:
