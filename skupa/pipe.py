#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import time


__all__ = ['Pipeline', 'Worker', 'Job']



class Pipeline:
    def __init__(self):
        self.workers = []
        self.clock = -1


    def add_worker(self, worker):
        self.workers.append(worker)
        worker.set_pipeline(self)


    async def start(self):
        order, unmet, duped, avail = solve_pipeline(self.workers)

        assert not unmet, \
            'Pipeline dependencies not satisfied: {deps}' \
                .format(deps=unmet)

        assert not duped, \
            'Pipeline has multiple sources of: {deps}' \
                .format(deps=duped)

        assert 'frame' in avail, \
            'Pipeline does not have a source element'

        self.workers = order

        for worker in self.workers:
            await worker.prepare()

        for worker in self.workers:
            await worker.start()


    def reset(self, hint=None):
        for worker in self.workers:
            worker.reset(hint)


    async def run(self):
        queue = []
        job = None

        while True:
            while len(queue) >= 60:
                old = queue.pop(0)
                await old.done()
                old.dispose()

            self.clock += 1
            job = Job(self.clock, job)

            for worker in self.workers:
                task = asyncio.create_task(worker._handle(job))
                job._tasks.append(task)

                for prov in worker.provides:
                    job._deps[prov] = task

            # Watch the last worker so that we know when the
            # job is finished and can start another one.
            queue.append(job)

            yield job


class Worker:
    requires = []
    provides = []
    after    = []

    def __init__(self):
        self.pipeline = None

    def set_pipeline(self, pipeline):
        self.pipeline = pipeline

    def reset(self, hint=None):
        pass

    async def prepare(self):
        pass

    async def start(self):
        pass

    async def _handle(self, job):
        for dep in self.requires:
            await job._deps[dep]

        for dep in self.after:
            if dep in job._deps:
                await job._deps[dep]

        return await self.process(job)

    async def process(self, job):
        raise NotImplementedError


class Job:
    def __init__(self, tick, prev):
        self.id = tick
        self.ts = None

        self.prev = prev

        self._tasks = []
        self._deps  = {}

    async def done(self):
        for task in self._tasks:
            await task

        return self

    def mark_start(self):
        assert self.ts is None, 'mark_start called repeatedly'
        self.ts = time.time()

    def __repr__(self):
        return 'Job(id={})'.format(self.id)

    def dispose(self):
        self.prev = None


def solve_pipeline(workers):
    workers = set(workers)

    order = []
    duped = set()
    unmet = set()
    avail = set()

    for worker in workers:
        avail.update(worker.provides)
        unmet.update(worker.requires)

    satisfied = set()

    while workers:
        removed = 0

        for worker in list(workers):
            deps  = set(worker.requires)
            deps |= set(worker.after) & avail

            if not deps - satisfied:
                removed += 1
                order.append(worker)
                workers.discard(worker)

                duped     |= satisfied & set(worker.provides)
                satisfied |= set(worker.provides)
                unmet     -= set(worker.provides)

        if not removed:
            break

    return order, unmet, duped, avail


# vim:set sw=4 ts=4 et:
