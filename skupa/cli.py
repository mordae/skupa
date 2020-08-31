#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import os


@click.group(chain=True, invoke_without_command=True)
@click.option('-r', '--rate', default=30, help='Output rate limit')
def main(**kw):
    pass


@main.command('camera', help='OpenCV-based camera input')
@click.option('-d', '--device', default=0, help='Device number')
def camera(device):
    from skupa.video.camera import CameraFeed
    return CameraFeed(device)


@main.command('dlib-face', help='DLib-based face detector')
def dlib_face():
    from skupa.face.dlib import FaceDetector
    return FaceDetector()


@main.command('onnx-face', help='ONNX-based face detector')
@click.option('--slim/--no-slim', default=True, help='Use lightweight model')
def onnx_face(slim):
    from skupa.face.onnx import FaceDetector
    return FaceDetector(slim)


@main.command('dlib-lms', help='DLib-based landmark detector')
@click.option('--tracking/--no-tracking', default=True,
              help='Use expensive visual flow tracking')
def dlib_lms(tracking):
    from skupa.lms.dlib import LandmarkDetector
    return LandmarkDetector(tracking)


@main.command('onnx-lms', help='ONNX-based landmark detector')
@click.option('--tracking/--no-tracking', default=True,
              help='Use expensive visual flow tracking')
def onnx_lms(tracking):
    from skupa.lms.onnx import LandmarkDetector
    return LandmarkDetector(tracking)


@main.command('lms-rpy', help='Landmark-based roll/pitch/yaw estimator')
@click.option('-r', '--roll', default=0, help='Roll correction')
@click.option('-p', '--pitch', default=0, help='Pitch correction')
@click.option('-y', '--yaw', default=0, help='Yaw correction')
def lms_face_pose(**kw):
    from skupa.head.lms import HeadPoseEstimator
    return HeadPoseEstimator(**kw)


@main.command('lms-eyes', help='Landmark-based eye tracking')
@click.option('--raw/--no-raw', default=False, help='Emit raw eye openness levels')
def lms_eyes(**kw):
    from skupa.eyes.lms import EyesTracker
    return EyesTracker(**kw)


@main.command('auto-eyes', help='Time-based automatic blinking')
@click.option('-i', '--interval', default=60, help='Blink every n-th second')
def auto_eyes(interval):
    from skupa.eyes.auto import EyesTracker
    return EyesTracker(interval)


@main.command('headband', help='Headbandroll/pitch/yaw reader')
@click.option('-d', '--device', default='/dev/ttyUSB0', help='Serial port')
@click.option('-r', '--roll', default=0, help='Roll correction')
@click.option('-p', '--pitch', default=0, help='Pitch correction')
@click.option('-y', '--yaw', default=0, help='Yaw correction')
def headband(**kw):
    from skupa.head.headband import HeadPoseReader
    return HeadPoseReader(**kw)


@main.command('audio-mouth', help='Audio-based mouth tracking')
@click.option('-l', '--language', default='cs', help='Language model to use')
def audio_mouth(language):
    from skupa.mouth.audio import AudioMouthTracker
    return AudioMouthTracker(language)


@main.command('osc-proto', help='OSC-based network sender')
@click.option('-h', '--host', default='localhost', help='Recipient host')
@click.option('-p', '--port', default=9001, help='Recipient port')
@click.option('-i', '--index', default=0, help='Face index to use')
def osc_proto(**kw):
    from skupa.proto.osc import OSCProtocol
    return OSCProtocol(**kw)


@main.command('json-proto', help='UDP/JSON-based network sender')
@click.option('-h', '--host', default='localhost', help='Recipient host')
@click.option('-p', '--port', default=9001, help='Recipient port')
@click.option('-i', '--index', default=0, help='Face index to use')
def osc_proto(**kw):
    from skupa.proto.json import JSONProtocol
    return JSONProtocol(**kw)


@main.command('preview', help='OpenCV-based preview')
@click.option('--face/--no-face', default=True, help='Show face box')
@click.option('--lms/--no-lms', default=True, help='Show landmarks')
@click.option('--rpy/--no-rpy', default=True, help='Show roll/pitch/yaw')
@click.option('--eyes/--no-eyes', default=True, help='Show eyes')
@click.option('--mouth/--no-mouth', default=True, help='Show mouth')
def preview(**kw):
    from skupa.preview import Preview
    return Preview(**kw)


@main.resultcallback()
def run_pipeline(workers, rate):
    from skupa.pipe import Pipeline
    from concurrent.futures import ThreadPoolExecutor
    from asyncio import run, get_running_loop

    if not workers:
        print('No workers configured.')
        print("Run `skupa --help' to get a list.")
        return

    meta = {}
    pipe = Pipeline(meta)

    for worker in workers:
        pipe.add_worker(worker)

    async def amain():
        loop = get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(32))

        await pipe.start()
        async for job in pipe.run(rate):
            # TODO: Maybe track latency?
            pass

    try:
        run(amain())
    except KeyboardInterrupt:
        print()
        os._exit(0)


if __name__ == '__main__':
    main()


# vim:set sw=4 ts=4 et:
