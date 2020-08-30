#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio


def defer(fn, *args):
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, fn, *args)


# vim:set sw=4 ts=4 et:
