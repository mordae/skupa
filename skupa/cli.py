#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import click
import cv2
import numpy as np
import time

import skupa.face
import skupa.head
import skupa.lms
import skupa.proto

from skupa.util import defer, create_task
from skupa.features import FeatureTracker


@click.command()
@click.option('-f', '--face-detector', default='dlib',
              help='Face detector to use (dlib, onnx)')

@click.option('-l', '--landmark-detector', default='dlib',
              help='Landmark detector to use (dlib, onnx)')

@click.option('-e', '--head-model', default='rpy',
              help='Head model to use (eos, rpy)')

@click.option('-v', '--view/--no-view', default=False,
              help='View diagnostic output')

@click.option('-c', '--camera', type=int, default=0,
              help='Number of the webcam to use')

@click.option('-r', '--rate', type=int, default=30,
              help='Maximum frame rate to allow')

@click.option('-y', '--yaw', default=0,
              help='Yaw adjustment (to compensate for camera angle)')

@click.option('-L', '--latency', default=500,
              help='Default frame output latency')

@click.option('-p', '--protocol', default='none',
              help='Network protocol to use (none, osc)')

@click.option('-H', '--host', default='localhost',
              help='Host to send data to')

@click.option('-P', '--port', type=int, default=9001,
              help='Port to send data to')

def main(face_detector, landmark_detector, head_model, view, camera, rate, yaw, latency, protocol, host, port):
    assert face_detector in skupa.face.detectors, \
           'Invalid face detector selected'

    assert landmark_detector in skupa.lms.detectors, \
           'Invalid landmark detector selected'

    assert head_model in skupa.head.models, \
           'Invalid head model selected'

    assert protocol in skupa.proto.protocols, \
           'Invalid network protocol selected'

    face_detector = skupa.face.detectors[face_detector]()
    landmark_detector = skupa.lms.detectors[landmark_detector]()

    if landmark_detector.HEIGHT > face_detector.HEIGHT:
        width  = landmark_detector.WIDTH
        height = landmark_detector.HEIGHT
    else:
        width  = face_detector.WIDTH
        height = face_detector.HEIGHT

    head_model = skupa.head.models[head_model](width, height)
    protocol = skupa.proto.protocols[protocol](host, port)

    cam = cv2.VideoCapture(camera)
    assert cam.isOpened(), 'Failed to open camera'

    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    # Scale latency to seconds.
    latency /= 1000

    pipe = pipeline(face_detector, landmark_detector, head_model, cam, rate, yaw, latency, protocol, view)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(pipe)
    print('Done.')


async def pipeline(fd, ld, hm, cam, rate, yaw, latency, proto, view):
    queue = asyncio.Queue()
    trackers = {}

    async def read_frames():
        while True:
            deadline = time.time() + 1.0 / rate - 0.005

            res, frame = await defer(cam.read)
            assert res, 'Failed to read camera frame'

            yield time.time(), frame

            idle = deadline - time.time()
            if idle > 0:
                await asyncio.sleep(idle)


    async def publish_frames():
        nonlocal latency

        fps = 0
        total = 0
        start = time.time()

        previous = 0

        while True:
            started, task = await queue.get()
            faces = await task

            deadline = previous + latency + 1.0 / rate - 0.005
            previous = started

            for frame, box, vertices, feat in faces:
                # Send the features across the network.
                proto.send(feat)

                # Preview if requested.
                if view:
                    h, w, _ = frame.shape

                    cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)

                    dims = np.abs(box[:2] - box[2:])
                    cv2.putText(frame, '%i x %i' % tuple(dims), tuple(box[0:2]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0))

                    for x, y in feat.lms:
                        try:
                            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
                        except:
                            pass

                    for x, y in np.int32(vertices):
                        try:
                            if len(vertices) > 10:
                                # Small dots if there are a lot of them.
                                cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)
                            else:
                                # Large circles if there are only few.
                                cv2.circle(frame, (x, y), 3, (255, 255, 255), 1)
                        except:
                            pass

                    # Roll / Pitch / Yaw indicators
                    cv2.putText(frame, 'R: %6.2f' % feat.rpy[0], (5, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
                    cv2.putText(frame, 'P: %6.2f' % feat.rpy[1], (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))
                    cv2.putText(frame, 'Y: %6.2f' % feat.rpy[2], (5, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

                    # FPS counter
                    cv2.putText(frame, '%2i fps' % fps, (w - 50, 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0))

                    # Eye status
                    re, le = np.int32(feat.eyes * 50)
                    cv2.rectangle(frame, (w // 2 - 50, h), (w // 2 + 50, h - 50), (0, 0, 0), -1)
                    cv2.rectangle(frame, (w // 2 - 50, h), (w // 2,      h - re), (255, 255, 255), -1)
                    cv2.rectangle(frame, (w // 2,      h), (w // 2 + 50, h - le), (255, 255, 255), -1)

                    # Mouth status
                    cv2.putText(frame, 'w=%.3f h=%.3f' % tuple(feat.mouth), (w // 2 - 250, h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))

                    expr_names = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise']
                    for i, name in enumerate(expr_names):
                        cv2.putText(frame, name, (10, 80 + i * 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, feat.expr[i], (0, 0, 0))

            if view:
                cv2.imshow('Debug View', frame)
                if cv2.waitKey(1) == ord('q'):
                    asyncio.get_event_loop().stop()

            if time.time() >= start + 1.0:
                fps, total = total, 0
                start = time.time()
            else:
                total += 1

            # Smooth out output rate.
            finished = time.time()

            if finished - started >= latency:
                latency = finished - started
                print('Latency is now', int(1000 * latency), 'ms')

            idle = deadline - finished

            if idle > 0:
                await asyncio.sleep(idle)


    async def process_frame(frame):
        boxes, probs = await fd.detect(frame)

        results = []

        for i, box in enumerate(boxes):
            # Extract face landmarks
            lms = np.int32(await ld.detect(frame, box))

            # Estimate head pose
            rpy, expr, vertices = await hm.fit(lms, view)

            # Compensate yaw due to the camera angle.
            rpy[2] += yaw

            # Get the feature tracker for this face.
            if i not in trackers:
                trackers[i] = FeatureTracker(i)

            # Estimate the features.
            feat = trackers[i].track(lms, rpy, expr)

            results.append((frame, box, vertices, feat))

        return results


    create_task(publish_frames())

    async for ts, frame in read_frames():
        queue.put_nowait((ts, create_task(process_frame(frame))))


if __name__ == '__main__':
    main()


# vim:set sw=4 ts=4 et:
