#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from oscpy.client import OSCClient

__all__ = ['protocols']


class NoProtocol:
    def __init__(self, host, port):
        pass

    def send(self, feat):
        pass


class OSCProtocol:
    def __init__(self, host, port):
        self.client = OSCClient(host, port)

    def _send(self, route, items):
        return self.client.send_message(route, items)

    def send(self, feat):
        rpy   = [float(x) for x in feat.rpy]
        eyes  = [float(x) for x in feat.eyes]
        mouth = [float(x) for x in feat.mouth]
        expr  = [float(x) for x in feat.expr]

        self._send(b'/face/%i/rpy'   % feat.index, rpy)
        self._send(b'/face/%i/eyes'  % feat.index, eyes)
        self._send(b'/face/%i/mouth' % feat.index, mouth)
        self._send(b'/face/%i/expr'  % feat.index, expr)


protocols = dict(
    none = NoProtocol,
    osc  = OSCProtocol,
)


# vim:set sw=4 ts=4 et:
