#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from oscpy.client import OSCClient

from skupa.pipe import Worker


__all__ = ['OSCProtocol']


class OSCProtocol(Worker):
    after = ['rpy', 'eyes', 'mouth']

    def __init__(self, host, port, index):
        self.host  = host
        self.port  = port
        self.index = index

    async def start(self):
        self.client = OSCClient(self.host, self.port)

    def _send(self, route, items):
        return self.client.send_message(route, items)

    async def process(self, job):
        if getattr(job, 'rpy', None) is not None:
            rpy = [float(x) for x in job.rpy]
            self._send(b'/face/%i/rpy' % self.index, rpy)

        if getattr(job, 'eyes', None) is not None:
            eyes = [float(x) for x in job.eyes]
            self._send(b'/face/%i/eyes' % self.index, eyes)

        if getattr(job, 'mouth', None) is not None:
            mouth = [float(x) for x in job.mouth]
            self._send(b'/face/%i/mouth' % self.index, mouth)


# vim:set sw=4 ts=4 et:
