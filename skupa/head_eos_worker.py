#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This module should not be imported directly.
It serves as a base for head_eos worker processes.
"""

import eos
import math
import numpy as np

import skupa.head_eos as h


shape    = eos.morphablemodel.load_model(h.SHAPE_PATH)
topology = eos.morphablemodel.load_edge_topology(h.TOPOLOGY_PATH)
contours = eos.fitting.ModelContour.load(h.CONTOURS_PATH)
expr_bs  = eos.morphablemodel.load_blendshapes(h.EXPR_BS_PATH)

model = eos.morphablemodel.MorphableModel(
    shape.get_shape_model(), expr_bs,
    color_model = eos.morphablemodel.PcaModel(),
    texture_coordinates = shape.get_texture_coordinates(),
)

landmark_mapper   = eos.core.LandmarkMapper(h.LMS_MAP_PATH)
contour_landmarks = eos.fitting.ContourLandmarks.load(h.LMS_MAP_PATH)


def fit(width, height, vm, lms, vertices):
    # Convert landmarks to the required format.
    landmarks = []

    for i, pt in enumerate(lms, 1):
        landmarks.append(eos.core.Landmark(str(i), pt))

    # Fit head model to the landmarks.
    mesh, pose, _coeff, expr = eos.fitting.fit_shape_and_pose(
        model, landmarks, landmark_mapper,
        width, height, topology,
        contour_landmarks, contours
    )

    yaw, pitch, roll = pose.get_rotation_euler_angles()
    rpy = np.float32([math.degrees(roll), math.degrees(pitch), math.degrees(yaw)])

    if vertices:
        pr = pose.get_projection()
        mv = pose.get_modelview()

        fm = vm @ pr @ mv

        points = []

        for i in mesh.vertices:
            tmp = fm @ np.append(i, 1)
            points.append((int(width / 2 + tmp[0]), int(height / 2 + tmp[1])))

        return rpy, expr, np.int32(points)

    return rpy, expr, np.int32([])


# vim:set sw=4 ts=4 et:
