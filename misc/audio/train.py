#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import wave
import pickle
import os.path

from scipy.signal import blackman, welch
from sklearn.neural_network import MLPClassifier
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


RATE = 44100

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

if not os.path.exists('data.pickle'):
    for mi, path in enumerate(files):
        print('learning', labels[mi], 'from', path)

        for sps in [30, 60]:
            print('- sps', sps)

            sample = RATE // sps

            for mul in [1/6, 1/3, 1.0, 3.0, 6.0]:
                print('- volume', mul)

                for roll in [-2, -1, 0, +1, +2]:
                    print('- pitch roll', roll)

                    wav = wave.open(path, 'rb')

                    while True:
                        chunk = np.frombuffer(wav.readframes(sample), dtype='<i2')

                        if len(chunk) < sample:
                            break

                        # Learn at different volume levels
                        chunk = chunk * mul

                        # Learn at different frequencies
                        chunk = np.roll(chunk, roll)

                        if roll > 0:
                            chunk[:roll] = 0
                        else:
                            chunk[roll:] = 0

                        # Add some normal noise
                        chunk += np.random.normal(0, 2, len(chunk))

                        volume = 20. * np.log10(np.max(np.abs(chunk)) / 32768)

                        freqs, densities = welch(chunk, RATE, nperseg=len(chunk))
                        freqs = freqs[:cutoff]
                        densities = densities[:cutoff]
                        densities = 20. * np.log10(densities)
                        densities = np.nan_to_num(densities)

                        # Add more noise to the densities.
                        densities += np.random.normal(0, 2, len(densities))

                        data.append(np.array([*densities, *prevd, volume]))
                        feat.append(mi)

                        prevd = 0.5 * prevd + 0.5 * densities

                    wav.close()

    with open('data.pickle', 'wb') as fp:
        print('Saving data.pickle...')
        pickle.dump((data, feat), fp)

else:
    print('Loading data.pickle...')
    with open('data.pickle', 'rb') as fp:
        data, feat = pickle.load(fp)


print('Fitting...')
model = MLPClassifier(hidden_layer_sizes=(60, 30, 24, 12),
                      #solver='sgd',
                      #learning_rate='adaptive',
                      #solver='lbfgs',
                      max_iter=10,
                      verbose=True)

model = model.fit(data, feat)

print('saving to model.onnx')

initial_type = [('float_input', FloatTensorType([None, 201]))]
model = convert_sklearn(model, initial_types=initial_type)
with open('model.onnx', 'wb') as fp:
    fp.write(model.SerializeToString())


# vim:set sw=4 ts=4 et:
