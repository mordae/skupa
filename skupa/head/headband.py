#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import cv2
import json
import numpy as np
import serial

from scipy.stats import circmean

from skupa.pipe import Worker
from skupa.util import defer


__all__ = ['HeadPoseReader']


class HeadPoseReader(Worker):
    requires = []
    provides = ['rpy']

    def __init__(self, device, roll, pitch, yaw):
        self.device = device
        self.correction = np.array([roll, pitch, yaw])

    async def start(self):
        # In radians for simpler averaging.
        self.rpy = np.zeros(3)

        # Device we are reading from.
        self.serial = serial.Serial(self.device, 115200)
        self.task = asyncio.create_task(self._read_rpy())

    async def _read_rpy(self):
        while True:
            line = await defer(self.serial.readline)

            if not line.startswith(b'RPY:'):
                continue

            rpy = json.loads(line[4:].strip())
            rpy = rpy[1], rpy[0], rpy[2]
            rpy = np.radians(rpy)

            for i in range(3):
                self.rpy[i] = circmean([self.rpy[i]] * 9 + [rpy[i]])

    async def process(self, job):
        job.rpy = np.degrees(self.rpy) + self.correction
        job.rpy[job.rpy > 180] -= 360


# vim:set sw=4 ts=4 et: