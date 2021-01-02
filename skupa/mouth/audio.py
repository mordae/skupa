#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import numpy as np
import onnxruntime as ort

from os.path import join, dirname, exists
from scipy.signal import welch

from skupa.pipe import Worker
from skupa.util import defer


__all__ = ['AudioMouthTracker']


MODEL_PATH = join(dirname(__file__), '..', 'model',
                  'audio', 'audio-{lang}.onnx')

RATE   = 44100
CUTOFF = 100
LABELS = ['A', 'E', 'I', 'O', 'U', '-']

# How quickly the mouth changes shape to incorporate target vowels.
VOWEL_ATTACK_RATE = .9

# How quickly do the vowels decay.
VOWEL_DECAY_RATE = .8


class AudioMouthTracker(Worker):
    provides = ['mouth']
    requires = ['frame', 'audio']

    def __init__(self, language):
        self.language = language
        path = MODEL_PATH.format(lang=self.language)
        assert exists(path), 'Unsupported audio mouth tracker language'


    async def start(self):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3

        path = MODEL_PATH.format(lang=self.language)
        self.session = ort.InferenceSession(path, sess_options=opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Individual audio buffers.
        self.adata = []

        # Current vowel weights.
        self.vowels  = np.zeros(len(LABELS))

        # Averaged vowel weights to smoothen mouth movement.
        self.average = np.zeros(len(LABELS))

        # Previous frequencies for smoothing.
        self.prev = np.zeros(CUTOFF)

        # Most recent volume.
        self.volume = 0.

    async def _read_frames(self):
        while True:
            frame = await defer(self.stream.read, SAMPLE_SIZE)
            frame = np.frombuffer(frame, dtype='<i2')
            self.frames.append(frame)

    async def process(self, job):
        if len(job.audio) > 0:
            self.adata.append(job.audio)

        frame_size = round(RATE / job.frame_rate)
        frame = np.zeros(frame_size, dtype='<i2')
        sofar = 0

        while sofar < frame_size:
            if not self.adata:
                print('Audio buffer underflow...')

                if sofar:
                    self.adata.insert(0, frame[:sofar])

                return

            chunk = self.adata.pop(0)
            needed = frame_size - sofar

            if len(chunk) < needed:
                frame[sofar : sofar + len(chunk)] = chunk
                sofar += len(chunk)

            else:
                frame[sofar:] = chunk[:needed]
                sofar += needed
                self.adata.insert(0, chunk[needed:])

        self.volume = 20. * np.log10(np.max(np.abs(frame)) / 32768)

        freqs, densities = welch(frame, job.frame_rate, nperseg=len(frame))
        freqs = freqs[:CUTOFF]

        densities = densities[:CUTOFF]
        densities = 20. * np.log10(densities)
        densities = np.nan_to_num(densities)

        # Decay weights over time.
        self.vowels = np.around(self.vowels * VOWEL_DECAY_RATE, 4)

        # Identify the vowel (or silence).
        inputs = [*densities, *self.prev, self.volume]
        res = await defer(self.session.run,
                          [self.output_name],
                          {self.input_name: [inputs]})

        # This is the vowel model heard the best.
        self.vowels[int(res[0])] = 1

        # Smooth out densities over time.
        self.prev = self.prev * .5 + densities * .5

        # Make vowels attack non-instantly.
        self.average = self.average * (1. - VOWEL_ATTACK_RATE) \
                     + self.vowels  *       VOWEL_ATTACK_RATE

        # Present the output.
        job.mouth = self.average[:5]

        # Also include some misc information.
        job.audio_volume = self.volume


# vim:set sw=4 ts=4 et:
