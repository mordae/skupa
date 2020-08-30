#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import asyncio

from skupa.pipe import Worker
from skupa.util import defer


__all__ = ['CameraFeed']


class CameraFeed(Worker):
    provides = ['frame']

    def __init__(self, device):
        self.device = device

    async def start(self):
        self.cam = cv2.VideoCapture(self.device)

        assert self.cam.isOpened(), \
            'Failed to open camera {}'.format(self.device)

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.meta.get('width', 320))
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.meta.get('height', 240))

        self.frame = None
        self.task = asyncio.create_task(self._read_frames())

    async def _read_frames(self):
        while True:
            res, frame = await defer(self.cam.read)

            if res:
                self.frame = frame
            else:
                print('Failed to read camera frame!')

    async def process(self, job):
        job.frame = self.frame


# vim:set sw=4 ts=4 et:
