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
from urllib.parse import quote


__all__ = ['PlaybackFeed']


class PlaybackFeed(Worker):
    provides = ['frame', 'audio']

    def __init__(self, path, video_rate=0):
        self.path = path
        self.video_rate = video_rate

    async def prepare(self):
        self.pipeline = Gst.ElementFactory.make('playbin3')

        if '://' in self.path:
            self.pipeline.set_property('uri', quote(self.path))
        else:
            quri = quote('file://' + abspath(self.path), ':/')
            self.pipeline.set_property('uri', quri)

        video_opts = ['video/x-raw', 'format=BGR']

        if self.video_rate:
            video_opts.append(f'framerate={self.video_rate}/1')

        self.pipeline.set_property('video-sink',
            Gst.parse_bin_from_description(f'''
                queue
                ! videorate
                ! videoconvert
                ! {','.join(video_opts)}
                ! videoconvert
                ! appsink name=video max-buffers=60
            ''', True))

        self.pipeline.set_property('audio-sink',
            Gst.parse_bin_from_description('''
                tee name=t
                ! queue
                ! autoaudiosink

                t.
                ! queue
                ! audioconvert
                ! audioresample
                ! audio/x-raw, format=S16LE, channels=1, rate=44100
                ! appsink name=audio max-buffers=60
            ''', True))

        self.videosink = self.pipeline.get_property('video-sink').get_by_name('video')
        self.audiosink = self.pipeline.get_property('audio-sink').get_by_name('audio')

        self.pipeline.set_state(Gst.State.PAUSED)


    async def start(self):
        self.pipeline.set_state(Gst.State.PLAYING)


    async def process(self, job):
        # Make sure that the previous job have successfully
        # grabbed its frame before us to guarantee a sequence.
        if job.prev is not None:
            await job.prev._deps['frame']

        frame = await defer(self.videosink.pull_sample)
        job.mark_start()

        samples = []
        while True:
            sample = self.audiosink.try_pull_sample(0)

            if sample is None:
                break

            samples.append(sample)

        if frame is None:
            print('No more video frames, exiting...')
            os._exit(0)

        caps = frame.get_caps()
        structure = caps.get_structure(0)

        ok, num, denom = structure.get_fraction('framerate')
        if not ok:
            print('Failed to determine video frame rate, exiting...')
            os._exit(1)

        job.frame_rate = num / denom

        width = structure.get_value('width')
        height = structure.get_value('height')

        fmt = structure.get_string('format')
        if fmt != 'BGR':
            print('Failed to get BGR frame from the video, exiting...')
            os._exit(1)

        vbuf = frame.get_buffer()

        try:
            ok, vmi = vbuf.map(Gst.MapFlags.READ)
            assert ok, 'Failed to map buffer data for READ'
            job.frame = np.ndarray(shape=(height, width, 3),
                                   dtype=np.uint8,
                                   buffer=vmi.data)
        finally:
            vbuf.unmap(vmi)

        adata = b''
        for sample in samples:
            abuf = sample.get_buffer()
            adata += abuf.extract_dup(0, abuf.get_size())

        job.audio = np.frombuffer(adata, dtype='<i2')


# vim:set sw=4 ts=4 et:
