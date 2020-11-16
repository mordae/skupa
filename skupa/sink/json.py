#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import socket
import json

from skupa.pipe import Worker


__all__ = ['JSONSink']


class JSONSink(Worker):
    after = ['rpy', 'eyes', 'mouth', 'emote']

    def __init__(self, host, port, index):
        self.host  = host
        self.port  = port
        self.index = index

    async def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def _send(self, route, items):
        return self.client.send_message(route, items)

    async def process(self, job):
        # Make sure that the previous job have finished to ensure
        # proper ordering.
        if job.prev is not None:
            await job.prev.done()

        data = {'index': self.index}

        if getattr(job, 'rpy', None) is not None:
            data['rpy'] = [round(float(x), 5) for x in job.rpy]

        if getattr(job, 'eyes', None) is not None:
            data['eyes'] = [round(float(x), 5) for x in job.eyes]

        if getattr(job, 'mouth', None) is not None:
            data['mouth'] = [round(float(x), 5) for x in job.mouth]

        if getattr(job, 'emote', None) is not None:
            data['emote'] = job.emote

        bstr = json.dumps(data).encode('utf8')
        self.socket.sendto(bstr, (self.host, self.port))


# vim:set sw=4 ts=4 et:
