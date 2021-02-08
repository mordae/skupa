#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import numpy as np
import os
import time

import gi
gi.require_version('Gst', '1.0')

from gi.repository import Gst
Gst.init(None)

from skupa.pipe import Worker

from os.path import abspath
from urllib.parse import quote


__all__ = ['ScriptSource']


class ScriptSource(Worker):
    provides = ['frame', 'rpy', 'mouth', 'eyes']

    def __init__(self, path, audio=None, offset=0):
        self.path   = path
        self.audio  = audio
        self.offset = offset


    async def prepare(self):
        self.frames = []
        self.rate = None

        if self.audio is not None:
            self._prepare_audio()

        with open(self.path, 'r') as fp:
            for line in fp.readlines():
                line = line.strip()

                if line.startswith('#'):
                    continue

                line = json.loads(line)

                if 'version' in line:
                    assert line['version'] == 1, \
                        'Unsupported script version, expected: 1'

                    self.rate = line['rate']

                else:
                    frame = {}

                    if 'rpy' in line:
                        frame['rpy'] = np.array(line['rpy'])

                    if 'mouth' in line:
                        frame['mouth'] = np.array(line['mouth'])

                    if 'eyes' in line:
                        frame['eyes'] = np.array(line['eyes'])

                    self.frames.append(frame)

            assert self.rate is not None, 'No framerate specified'


    def _prepare_audio(self):
        self.pipeline = Gst.ElementFactory.make('playbin3')

        if '://' in self.path:
            self.pipeline.set_property('uri', quote(self.audio))
        else:
            quri = quote('file://' + abspath(self.audio), ':/')
            self.pipeline.set_property('uri', quri)
            self.pipeline.set_property('av-offset', self.offset)

        self.pipeline.set_property('video-sink',
            Gst.parse_bin_from_description('fakesink', True))

        self.pipeline.set_state(Gst.State.PAUSED)


    def start(self):
        if self.audio is not None:
            self.pipeline.set_state(Gst.State.PLAYING)


    async def process(self, job):
        # Make sure that the previous job have successfully
        # grabbed its frame before us to guarantee a sequence.

        if job.prev is not None:
            await job.prev._deps['frame']

        try:
            frame = self.frames[job.id]
        except IndexError:
            print('Ran out of script frames')
            os._exit(0)

        job.mark_start()

        if job.prev is None:
            self.start = time.time()
        else:
            await asyncio.sleep(self.start + job.id / self.rate - time.time())

        job.frame_rate = self.rate
        job.frame = np.full((480, 640, 3), 127, dtype=np.uint8)

        for k, v in frame.items():
            setattr(job, k, v)


# vim:set sw=4 ts=4 et:
