#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np


__all__ = ['HeadModel']


OBJECT_PTS = np.float32([[6.8258970, 6.760612, 4.402142],
                         [1.3303530, 7.122144, 6.903745],
                         [-1.330353, 7.122144, 6.903745],
                         [-6.825897, 6.760612, 4.402142],
                         [5.3114320, 5.485328, 3.987654],
                         [1.7899300, 5.393625, 4.413414],
                         [-1.789930, 5.393625, 4.413414],
                         [-5.311432, 5.485328, 3.987654],
                         [2.0056280, 1.409845, 6.165652],
                         [-2.005628, 1.409845, 6.165652]])

class HeadModel:
    def __init__(self, width, height):
        self.width  = width
        self.height = height

    async def fit(self, lms, vertices=False, **kw):
        w, h = self.width, self.height

        cam_matrix = np.float32([[ w, 0., w // 2],
                                 [0.,  w, h // w],
                                 [0., 0.,     1.]])

        dist_coeffs = np.float32([0., 0., 0., 0., 0.])

        image_pts = np.float32([lms[17], lms[21], lms[22], lms[26], lms[36],
                                lms[39], lms[42], lms[45], lms[31], lms[35]])

        _, rotation_vec, translation_vec = cv2.solvePnP(OBJECT_PTS, image_pts, cam_matrix, dist_coeffs)
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, (yaw, pitch, roll) = cv2.decomposeProjectionMatrix(pose_mat)

        rpy = np.float32([roll[0], pitch[0], yaw[0]])

        points = image_pts if vertices else np.float32([])
        return rpy, np.float32([0.] * 6), points


# vim:set sw=4 ts=4 et:
