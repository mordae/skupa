#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import numpy as np
import onnxruntime as ort
import pyaudio

from os.path import join, dirname, exists
from scipy.signal import blackman, welch

from skupa.pipe import Worker
from skupa.util import defer


__all__ = ['AudioMouthTracker']


MODEL_PATH = join(dirname(__file__), '..', 'model',
                  'audio', 'audio-{lang}.onnx')

RATE  = 44100
SPS   = 60

SAMPLE_SIZE = RATE // SPS
CUTOFF      = 100

LABELS = ['A', 'E', 'I', 'O', 'U', '-']


# How many dB above the floor is it still noise?
NOISE_BAND = 2.

# Speed of noise floor rising.
# Must be significant but allow for a drop during moments of silence.
NOISE_CLIMB = .0001

# Aggresivity of noise profile update.
# Must be high enough to compensate for NOISE_CLIMB during moments of silence.
NOISE_RATIO = .001

# How quickly do vowels decay.
# That is, how quickly does the mouth close after each sound.
VOWEL_DECAY_RATE = 0.5


class AudioMouthTracker(Worker):
    provides = ['mouth']

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

        self.frames = []
        self.task = asyncio.create_task(self._read_frames())

        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=RATE,
                                      input=True,
                                      frames_per_buffer=(SAMPLE_SIZE * 2))

        # Current vowel weights.
        self.vowels = np.zeros(len(LABELS))

        # Previous frequencies for smoothing.
        self.prev = np.zeros(CUTOFF)

        # Noise profile for noise reduction.
        self.noise = np.zeros(CUTOFF)

        # Start really high and drop to make sure we start filtering quickly.
        self.noise_floor = 0.

    async def _read_frames(self):
        while True:
            frame = await defer(self.stream.read, SAMPLE_SIZE)
            frame = np.frombuffer(frame, dtype='<i2')
            self.frames.append(frame)

    async def process(self, job):
        while len(self.frames) > 0:
            frame = self.frames.pop(0)

            window = blackman(len(frame))
            volume = 20. * np.log10(np.max(np.abs(frame)) / 32768)

            freqs, densities = welch(frame, RATE, nperseg=len(frame))
            freqs = freqs[:CUTOFF]

            densities = densities[:CUTOFF]
            densities = 20. * np.log10(densities)
            densities = np.nan_to_num(densities)

            self.noise_floor += NOISE_CLIMB

            if self.noise_floor < -120:
                self.noise_floor = 0.

            if np.abs(np.max(self.noise)) > 30:
                self.noise *= 0.

            if volume < self.noise_floor + NOISE_BAND:
                self.noise_floor = self.noise_floor * 0.99 + volume * 0.01
                self.noise = self.noise * (1. - NOISE_RATIO) \
                           + densities  * NOISE_RATIO

            densities -= self.noise
            densities[densities < 0] = 0

            # Decay weights over time.
            self.vowels *= VOWEL_DECAY_RATE

            if volume > self.noise_floor + 6 * NOISE_BAND:
                # Identify the vowel (or silence).
                inputs = [*densities, *self.prev, volume]
                res = await defer(self.session.run,
                                  [self.output_name],
                                  {self.input_name: [inputs]})

                # This is the vowel model heard the best.
                vowel = int(res[0])

                # TODO: Maybe remove this adjustments and leave
                #       them to the recipient?

                # More is too pronounced, so add just this much.
                self.vowels[vowel] = [.7, .5, .7, .7, .7, 1.][vowel]

                # Open mouth a bit for other vowels to be seen
                if vowel != 5 and self.vowels[0] < 0.4:
                    self.vowels[0] = 0.4

            # Smooth out densities over time.
            self.prev = self.prev * 0.5 + densities * 0.5

            # Present the output.
            job.mouth = self.vowels[:5]

            # Also include some misc information.
            job.audio_volume      = volume
            job.audio_noise_floor = self.noise_floor
            job.audio_denoise     = np.max(self.noise)


# vim:set sw=4 ts=4 et:
