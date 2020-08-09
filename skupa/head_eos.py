#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, dirname

import asyncio
import concurrent.futures
import cv2
import math
import numpy as np

__all__ = ['HeadModel']


SHAPE_PATH     = join(dirname(__file__), 'model', 'eos', 'sfm_shape_3448.bin')
TOPOLOGY_PATH  = join(dirname(__file__), 'model', 'eos', 'sfm_3448_edge_topology.json')
CONTOURS_PATH  = join(dirname(__file__), 'model', 'eos', 'sfm_model_contours.json')
EXPR_BS_PATH   = join(dirname(__file__), 'model', 'eos', 'expression_blendshapes_3448.bin')
LMS_MAP_PATH   = join(dirname(__file__), 'model', 'eos', 'ibug_to_sfm.txt')


class HeadModel:
    def __init__(self, width, height, max_workers=None):
        # The library is not implicitly parallel,
        # so we need to take care of that by ourselves.
        self.pool = concurrent.futures.ProcessPoolExecutor(max_workers)

        self.width  = width
        self.height = height

        viewport = np.array([0, height, width, -height])

        s = np.identity(4, dtype=np.float32)
        s[0][0] *= viewport[2] / 2
        s[1][1] *= viewport[3] / 2
        s[2][2] *= 0.5

        t = np.identity(4, dtype=np.float32)
        t[3][0] = viewport[0] + (viewport[2] / 2)
        t[3][1] = viewport[1] + (viewport[3] / 2)
        t[3][2] = 0.5

        self.viewport_matrix = s @ t


    async def fit(self, lms, vertices=False, **kw):
        loop = asyncio.get_event_loop()

        return await loop.run_in_executor(
                self.pool, worker,
                self.width, self.height, self.viewport_matrix,
                lms, vertices
        )


def worker(*args):
    import skupa.head_eos_worker
    return skupa.head_eos_worker.fit(*args)


# vim:set sw=4 ts=4 et:
