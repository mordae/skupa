#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import numpy as np

from skupa.pipe import Worker


__all__ = ['EyesTracker']


class EyesTracker(Worker):
    provides = ['eyes']

    def __init__(self, interval):
        self.interval = interval

    async def process(self, job):
        job.eyes = np.ones(2)

        fnow = time.time()
        inow = int(fnow)

        if 0 != inow % self.interval:
            return

        # Average blink takes 100-150 ms.
        # Make it a little bit cartoonish with faster drop and slower rise.
        job.eyes = np.interp([fnow - inow] * 2, [.0, .050, .150], [1., 0., 1.])


# vim:set sw=4 ts=4 et:
