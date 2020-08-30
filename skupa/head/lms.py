#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from skupa.pipe import Worker


__all__ = ['HeadPoseEstimator']


POINTS = np.float32([[ 6.8258970, 6.760612, 4.402142],
                     [ 1.3303530, 7.122144, 6.903745],
                     [-1.3303530, 7.122144, 6.903745],
                     [-6.8258970, 6.760612, 4.402142],
                     [ 5.3114320, 5.485328, 3.987654],
                     [ 1.7899300, 5.393625, 4.413414],
                     [-1.7899300, 5.393625, 4.413414],
                     [-5.3114320, 5.485328, 3.987654],
                     [ 2.0056280, 1.409845, 6.165652],
                     [-2.0056280, 1.409845, 6.165652]])


class HeadPoseEstimator(Worker):
    requires = ['frame', 'lms']
    provides = ['rpy']

    async def start(self):
        self.average = np.zeros(3)

    async def process(self, job):
        if job.lms is None or job.frame is None:
            job.rpy = None
            return

        h, w, _ = job.frame.shape

        dist_coeffs = np.zeros(5)
        cam_matrix = np.float32([[ w, 0., w // 2],
                                 [0.,  w, h // w],
                                 [0., 0.,     1.]])

        lms = job.lms
        points = np.float32([lms[17], lms[21], lms[22], lms[26], lms[36],
                             lms[39], lms[42], lms[45], lms[31], lms[35]])

        _, rotation_vec, translation_vec = cv2.solvePnP(POINTS, points, cam_matrix, dist_coeffs)
        rotation_mat, _ = cv2.Rodrigues(rotation_vec)
        pose_mat = cv2.hconcat((rotation_mat, translation_vec))
        _, _, _, _, _, _, (pitch, yaw, roll) = cv2.decomposeProjectionMatrix(pose_mat)

        rpy = np.array([roll[0], pitch[0], yaw[0]])

        if not self.average.any():
            self.average = rpy

        self.average = self.average * 0.5 + rpy * 0.5
        job.rpy = self.average


# vim:set sw=4 ts=4 et:
