#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave

from scipy.signal import blackman, welch
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


RATE = 44100
SAMPLE = RATE // 60

cutoff = 100

labels = ['A', 'E', 'I', 'O', 'U', '-']

files = [
    'data/A.wav',
    'data/E.wav',
    'data/I.wav',
    'data/O.wav',
    'data/U.wav',
    'data/silence.wav',
]

data = []
feat = []

prevd = np.zeros(cutoff)

for mi, path in enumerate(files):
    print('learning', labels[mi], 'from', path)

    for mul in [0.5, 0.7, 0.85, 1.0, 1.1, 1.25, 1.5, 2.0]:
        for roll in [-2, -1, 0, +1, +2]:
            wav = wave.open(path, 'rb')

            while True:
                chunk = np.frombuffer(wav.readframes(SAMPLE), dtype='<i2')

                if len(chunk) == 0:
                    break

                if len(chunk) < SAMPLE:
                    chunk = np.float64(list(chunk) + [0] * (SAMPLE - len(chunk)))

                # Learn at different volume levels
                chunk = chunk * mul

                volume = 20. * np.log10(np.max(np.abs(chunk)) / 32768)

                freqs, densities = welch(chunk, RATE, nperseg=len(chunk))
                freqs = freqs[:cutoff]
                densities = densities[:cutoff]
                densities = 20. * np.log10(densities)
                densities = np.nan_to_num(densities)

                # Learn at different frequencies
                chunk = np.roll(chunk, roll)

                # Add some normal noise
                chunk += np.random.normal(0, 2, len(chunk))

                if roll > 0:
                    chunk[:roll] = 0
                else:
                    chunk[roll:] = 0

                data.append([*densities, *prevd, volume])
                feat.append(mi)

                prevd = 0.5 * prevd + 0.5 * densities

            wav.close()

print('fitting')
model = MLPClassifier(hidden_layer_sizes=(101, 24),
                      solver='sgd',
                      #solver='lbfgs',
                      alpha=0.0001,
                      max_iter=1000,
                      learning_rate='adaptive',
                      verbose=True)

model = model.fit(data, feat)

print('saving to model.onnx')

initial_type = [('float_input', FloatTensorType([None, 201]))]
model = convert_sklearn(model, initial_types=initial_type)
with open('model.onnx', 'wb') as fp:
    fp.write(model.SerializeToString())


# vim:set sw=4 ts=4 et:
