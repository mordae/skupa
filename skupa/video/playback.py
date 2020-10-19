#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import numpy as np
import cv2
import os

import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstApp', '1.0')

from gi.repository import Gst, GstApp
Gst.init(None)

from skupa.pipe import Worker
from skupa.util import defer

from os.path import abspath


__all__ = ['PlaybackFeed']


class PlaybackFeed(Worker):
    provides = ['frame']

    def __init__(self, path):
        self.path = path

    async def start(self):
        self.pipeline = Gst.ElementFactory.make('playbin3', 'playbin3')

        if '://' in self.path:
            self.pipeline.set_property('uri', self.path)
        else:
            self.pipeline.set_property('uri', 'file://' + abspath(self.path))

        videocaps = Gst.Caps.new_empty_simple('video/x-raw')
        videocaps.set_value('format', 'RGB')

        self.videosink = GstApp.AppSink()
        self.videosink.set_max_buffers(60)
        self.videosink.set_property('caps', videocaps)
        self.pipeline.set_property('video-sink', self.videosink)

        self.audiosink = Gst.ElementFactory.make('fakeaudio', 'fakeaudio')
        self.pipeline.set_property('audio-sink', self.audiosink)

        self.pipeline.set_state(Gst.State.PLAYING)


    async def process(self, job):
        sample = await defer(self.videosink.pull_sample)

        if sample is None:
            print('No more video frames, exiting...')
            os._exit()

        caps = sample.get_caps()
        structure = caps.get_structure(0)

        ok, num, denom = structure.get_fraction('framerate')
        if not ok:
            print('Failed to determine video frame rate, exiting...')
            os._exit()

        job.frame_rate = num / denom

        width = structure.get_value('width')
        height = structure.get_value('height')

        fmt = structure.get_string('format')
        if fmt != 'RGB':
            print('Failed to get RGB frame from the video, exiting...')
            os._exit()

        buf = sample.get_buffer()
        data = buf.extract_dup(0, buf.get_size())
        job.frame = np.frombuffer(data, np.uint8).reshape(height, width, 3)
        job.frame = cv2.cvtColor(job.frame, cv2.COLOR_RGB2BGR)


# vim:set sw=4 ts=4 et:
