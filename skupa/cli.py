#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click
import os


@click.group(chain=True, invoke_without_command=True)
def main(**kw):
    pass


@main.command('live', help='GStreamer-based live audio/video input')
def live():
    from skupa.source.live import LiveFeed
    return LiveFeed()


@main.command('playback', help='GStreamer-based audio/video playback')
@click.option('-f', '--path', help='File path', required=True)
def playback(path):
    from skupa.source.playback import PlaybackFeed
    return PlaybackFeed(path)


@main.command('script', help='Replay script without any A/V')
@click.option('-f', '--path', help='Script path', required=True)
@click.option('-a', '--audio-path', help='Audio file path')
@click.option('-o', '--audio-offset', help='Audio offset', default=0)
def script(path, audio_path=None, audio_offset=0):
    from skupa.source.script import ScriptSource
    return ScriptSource(path, audio_path, audio_offset)


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


@main.command('audio-mouth', help='Audio-based mouth tracking')
@click.option('-l', '--language', default='cs', help='Language model to use')
def audio_mouth(language):
    from skupa.mouth.audio import AudioMouthTracker
    return AudioMouthTracker(language)


@main.command('ann-srt', help='SRT-based annotations')
@click.option('-f', '--path', help='Path to the subtitles', required=True)
def ann_srt(path):
    from skupa.ann.srt import AnnotateFromSRT
    return AnnotateFromSRT(path)


@main.command('json-sink', help='UDP/JSON-based network sender')
@click.option('-h', '--host', default='localhost', help='Recipient host')
@click.option('-p', '--port', default=9001, help='Recipient port')
@click.option('-i', '--index', default=0, help='Face index to use')
def json_sink(**kw):
    from skupa.sink.json import JSONSink
    return JSONSink(**kw)


@main.command('script-sink', help='Sink script to a file for later')
@click.option('-f', '--path', help='File path', required=True)
def script_sink(**kw):
    from skupa.sink.script import ScriptSink
    return ScriptSink(**kw)


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
def run_pipeline(workers):
    from skupa.pipe import Pipeline
    from concurrent.futures import ThreadPoolExecutor
    from asyncio import run, get_running_loop

    if not workers:
        print('No workers configured.')
        print("Run `skupa --help' to get a list.")
        return

    pipe = Pipeline()

    for worker in workers:
        pipe.add_worker(worker)

    async def amain():
        loop = get_running_loop()
        loop.set_default_executor(ThreadPoolExecutor(32))

        def handler(loop, context):
            loop.default_exception_handler(context)
            os._exit(2)

        loop.set_exception_handler(handler)

        await pipe.start()
        async for job in pipe.run():
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
