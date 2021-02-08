#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from scipy.spatial.distance import euclidean as distance

from skupa.pipe import Worker


__all__ = ['EyesTracker']


class EyesTracker(Worker):
    requires = ['lms']
    provides = ['eyes']

    def __init__(self, raw):
        self.raw = raw

    async def prepare(self):
        # Extreme open and closed eye measurements
        self.eyemin = np.float32([99, 99])
        self.eyemax = np.float32([00, 00])

    async def process(self, job):
        if job.lms is None:
            job.eyes = None
            return

        lms = job.lms

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
        out = np.float32([1., 1.])

        for i in range(2):
            if eyes[i] < self.eyemin[i]:
                self.eyemin[i] = max(0, eyes[i])

            if eyes[i] > self.eyemax[i]:
                self.eyemax[i] = max(0, eyes[i])

            r = [self.eyemin[i], self.eyemax[i]]
            out[i] = np.interp(eyes[i], r, [0., 1.])

        if self.raw:
            job.eyes = out
        else:
            if np.mean(out) > 0.5:
                job.eyes = np.array([1., 1.])
            else:
                job.eyes = np.array([0., 0.])


# vim:set sw=4 ts=4 et:
