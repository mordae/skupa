#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import socket
import json

from skupa.pipe import Worker


__all__ = ['ScriptSink']


class ScriptSink(Worker):
    requires = ['rpy', 'eyes', 'mouth']
    after    = ['emote']

    def __init__(self, path):
        self.path = path
        self.rate = None

    async def start(self):
        self.fp = open(self.path, 'w')

    async def process(self, job):
        # Make sure that the previous job have finished to ensure
        # proper ordering.
        if job.prev is not None:
            await job.prev.done()

        if self.rate is None:
            self.rate = job.frame_rate
            self.fp.write('# Skupa Script\n')
            json.dump({'version': 1, 'rate': self.rate}, self.fp)
            self.fp.write('\n')
            self.fp.flush()

        data = {}

        if getattr(job, 'rpy', None) is not None:
            data['rpy'] = [round(float(x), 3) for x in job.rpy]

        if getattr(job, 'eyes', None) is not None:
            data['eyes'] = [round(float(x), 3) for x in job.eyes]

        if getattr(job, 'mouth', None) is not None:
            data['mouth'] = [round(float(x), 3) for x in job.mouth]

        if getattr(job, 'emote', None) is not None:
            data['emote'] = job.emote

        if getattr(job, 'trigger', None) is not None:
            data['trigger'] = job.trigger

        json.dump(data, self.fp)
        self.fp.write('\n')
        self.fp.flush()


# vim:set sw=4 ts=4 et:
