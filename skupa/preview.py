#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import time

from skupa.pipe import Worker


PREVIEW_WIDTH  = 640
PREVIEW_HEIGHT = 480

FONT  = cv2.FONT_HERSHEY_SIMPLEX

WHITE = (255, 255, 255)
BLUE  = (255,   0,   0)
GREEN = (  0, 255,   0)
RED   = (  0,   0, 255)
BLACK = (  0,   0,   0)


class Preview(Worker):
    after = ['frame', 'face', 'lms', 'rpy', 'eyes', 'mouth']

    def __init__(self, face, lms, rpy, eyes, mouth):
        self.face  = face
        self.lms   = lms
        self.rpy   = rpy
        self.eyes  = eyes
        self.mouth = mouth

        self.latency = 0

    async def process(self, job):
        h, w = job.frame.shape[:2]

        if getattr(job, 'frame_rate', None) is not None:
            cv2.putText(job.frame, '%3.1f fps' % job.frame_rate, (w - 60, 10),
                        FONT, 0.4, BLACK)

        now = time.time()

        if self.latency < now - job.ts:
            self.latency = now - job.ts

        cv2.putText(job.frame, '%3i ms' % (self.latency * 1000),
                    (w - 60, 25), FONT, 0.4, BLACK)

        if self.face and getattr(job, 'face', None) is not None:
            cv2.rectangle(job.frame,
                          tuple(job.face[:2]), tuple(job.face[2:]),
                          GREEN, 1)

        if self.lms and getattr(job, 'lms', None) is not None:
            for x, y in np.int32(job.lms):
                try:
                    cv2.circle(job.frame, (x, y), 1, RED, -1)
                except KeyboardInterrupt:
                    raise
                except:
                    pass

        if self.rpy and getattr(job, 'rpy', None) is not None:
            rpy = job.rpy
            cv2.putText(job.frame, 'R: %6.2f' % rpy[0], (5, 10),
                        FONT, 0.4, BLACK)
            cv2.putText(job.frame, 'P: %6.2f' % rpy[1], (5, 20),
                        FONT, 0.4, BLACK)
            cv2.putText(job.frame, 'Y: %6.2f' % rpy[2], (5, 30),
                        FONT, 0.4, BLACK)

        if self.eyes and getattr(job, 'eyes', None) is not None:
            re, le = np.int32(job.eyes * 50)
            cv2.rectangle(job.frame, (w // 2 - 50, h), (w // 2 + 50, h - 50), BLACK, -1)
            cv2.rectangle(job.frame, (w // 2 - 50, h), (w // 2,      h - re), WHITE, -1)
            cv2.rectangle(job.frame, (w // 2,      h), (w // 2 + 50, h - le), WHITE, -1)

        if self.mouth and getattr(job, 'mouth', None) is not None:
            labels = ['A', 'E', 'I', 'O', 'U']

            for i, vowel in enumerate(job.mouth):
                cv2.putText(job.frame, labels[i],
                            (30 + 20 * i, h - 20), FONT, vowel, BLACK)

            if getattr(job, 'audio_volume', None) is not None:
                cv2.putText(job.frame, '%5.2f dB' % job.audio_volume,
                            (20, h - 45), FONT, 0.4, BLACK)

        cv2.imshow('Skupa Preview', job.frame)
        key = cv2.waitKey(1)

        if key == ord('q'):
            os._exit(0)
        elif key == ord('r'):
            print('Resetting rpy')
            self.pipeline.reset('rpy')


# vim:set sw=4 ts=4 et:
