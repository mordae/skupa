#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import datetime
import srt

from skupa.pipe import Worker


__all__ = ['AnnotateFromSRT']


EMOTES = {'angry', 'fun', 'joy', 'sorrow', 'surprised'}


class AnnotateFromSRT(Worker):
    requires = ['frame']
    provides = ['emote']

    def __init__(self, path):
        self.path = path

    async def start(self):
        with open(self.path, 'r') as fp:
            self.srt = list(srt.parse(fp.read(), False))

    async def process(self, job):
        job.emote = None

        now = datetime.timedelta(seconds=job.id / job.frame_rate)

        for sub in self.srt:
            if sub.start <= now < sub.end:
                for word in sub.content.split():
                    if word in EMOTES:
                        job.emote = word
                    else:
                        print('WARNING:', 'unknown annotation:', word)


# vim:set sw=4 ts=4 et:
