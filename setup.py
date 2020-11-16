#!/usr/bin/python3 -tt
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(**{
    'name': 'skupa',
    'version': '0.1.0',
    'author': 'Jan Hamal Dvořák',
    'description': 'Virtual puppet toolkit for VTubers',
    'license': 'MIT',
    'keywords': ['vtuber', 'pose estimation', 'lip-sync'],
    'url': 'http://github.com/mordae/skupa/',
    'include_package_data': True,
    'packages': find_packages(),
    'classifiers': [
        'License :: OSI Approved :: MIT License',
    ],
    'entry_points': '''
        [console_scripts]
        skupa=skupa.cli:main
    ''',
    'install_requires': [
        'click         >= 7.1',
        'dlib          >= 19.20',
        'numpy         >= 1.17',
        'onnxruntime   >= 1.4',
        'opencv-python >= 4.2',
        'oscpy         >= 0.5',
        'PyAudio       >= 0.2',
        'pyserial      >= 3.4',
        'scipy         >= 1.4',
        'PyGObject     >= 3.36',
        'srt           >= 3.4',
    ],
    'zip_safe': False,
})


# vim:set sw=4 ts=4 et:
