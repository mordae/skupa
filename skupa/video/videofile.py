#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import cv2
import os
import time

from skupa.pipe import Worker
from skupa.util import defer


__all__ = ['VideoFeed']


class VideoFeed(Worker):
    provides = ['frame']

    def __init__(self, path):
        self.path = path

    async def start(self):
        self.src = cv2.VideoCapture(self.path)

        assert self.src.isOpened(), \
            'Failed to open file {!r}'.format(self.path)

        self.start = time.time()
        self.rate  = self.src.get(cv2.CAP_PROP_FPS)
        self.num   = 0

        self.frame = None
        self.task = asyncio.create_task(self._read_frames())

    async def _read_frames(self):
        while True:
            self.num += 1
            target = self.start + self.num * (1 / self.rate)
            await asyncio.sleep(target - time.time())

            res, frame = await defer(self.src.read)

            if res:
                self.frame = frame
            else:
                os._exit(0)

    async def process(self, job):
        job.frame = self.frame
        job.frame_rate = self.rate


# vim:set sw=4 ts=4 et:
