#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import cv2
import numpy as np


def defer(fn, *args):
    loop = asyncio.get_running_loop()
    return loop.run_in_executor(None, fn, *args)


def resize_and_pad(src, size, color=127):
    if isinstance(size, int):
        size = (size, size)

    axes = list(src.shape)
    axes[:2] = size

    dst = np.full(axes, color, dtype=np.uint8)

    h, w = src.shape[:2]
    r = min(size[0] / h, size[1]/ w)
    h, w = round(h * r), round(w * r)

    dst[0:h, 0:w] = cv2.resize(src, (w, h))

    return dst, r


# vim:set sw=4 ts=4 et:
