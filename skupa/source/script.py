#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import json
import numpy as np
import os
import time

from skupa.pipe import Worker


__all__ = ['ScriptSource']


class ScriptSource(Worker):
    provides = ['frame', 'rpy', 'mouth', 'eyes']

    def __init__(self, path):
        self.path = path

    async def start(self):
        self.frames = []
        self.rate = None

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
