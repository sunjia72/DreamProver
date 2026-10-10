# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Small asyncio pool with tuple-aware success and fully awaited cancellation."""

import asyncio
import logging
from dreamprover.runtime.tracking import MaxLLMCallsExceeded
from dreamprover.runtime.budget import RunStopped
from dreamprover.lean.proof import VerificationInfrastructureError

logger = logging.getLogger(__name__)

FATAL_EXCEPTIONS = (MaxLLMCallsExceeded, VerificationInfrastructureError, RunStopped)


def raise_if_fatal(result):
    """Keep interruptions out of ordinary unsuccessful proof results."""
    if isinstance(result, FATAL_EXCEPTIONS):
        raise result


class AsyncJobPool:
    def __init__(self, max_concurrent=None):
        if max_concurrent is not None and max_concurrent < 1:
            raise ValueError("max_concurrent must be positive")
        self.tasks = []
        self.semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

    @staticmethod
    def _is_failure(result):
        if isinstance(result, BaseException):
            return True
        return not (result[0] if isinstance(result, tuple) and result else result)

    @staticmethod
    def _check_completed_interruptions(tasks):
        # Check the whole completed batch before returning an ordinary result.
        # Set iteration order must not hide an interrupted verifier or budget.
        for task in tasks:
            if not task.cancelled():
                error = task.exception()
                raise_if_fatal(error)
                if error is None:
                    raise_if_fatal(task.result())

    async def _run(self, coro, args, kwargs):
        if self.semaphore is None:
            return await coro(*args, **kwargs)
        async with self.semaphore:
            return await coro(*args, **kwargs)

    def submit(self, coro, *args, **kwargs):
        name = kwargs.pop("name", None)
        task = asyncio.create_task(self._run(coro, args, kwargs), name=name)
        self.tasks.append(task)
        return task

    async def _cleanup(self):
        tasks, self.tasks = self.tasks, []
        for task in tasks:
            if not task.done():
                task.cancel()
        # Await cancellation so workers finish their finally blocks before their
        # HTTP clients are closed. Also consume already finished exceptions.
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def wait_for_first_truthy(self):
        pending = set(self.tasks)
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                self._check_completed_interruptions(done)
                for task in done:
                    try:
                        result = task.result()
                    except FATAL_EXCEPTIONS:
                        raise
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        logger.exception("Job %s failed", task.get_name())
                        continue
                    if not self._is_failure(result):
                        return result
            return None
        finally:
            await self._cleanup()

    async def wait_for_all(self):
        tasks = list(self.tasks)
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                raise_if_fatal(result)
            return [(task.get_name(), result) for task, result in zip(tasks, results)]
        finally:
            await self._cleanup()

    async def wait_for_all_successful(self):
        return [(name, result) for name, result in await self.wait_for_all()
                if not isinstance(result, BaseException)]

    async def wait_until_first_failure_or_all_success(self):
        pending, successful = set(self.tasks), []
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                self._check_completed_interruptions(done)
                for task in done:
                    try:
                        result = task.result()
                    except FATAL_EXCEPTIONS:
                        raise
                    except (Exception, asyncio.CancelledError) as exc:
                        return [(task.get_name(), exc)]
                    if self._is_failure(result):
                        return [(task.get_name(), result)]
                    successful.append((task.get_name(), result))
            return successful
        finally:
            await self._cleanup()

    async def wait_for_first_failure_after_successes(self):
        pending, results = set(self.tasks), {}
        try:
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                self._check_completed_interruptions(done)
                for task in done:
                    try:
                        results[task] = task.result()
                    except FATAL_EXCEPTIONS:
                        raise
                    except (Exception, asyncio.CancelledError) as exc:
                        results[task] = exc
                for task in self.tasks:
                    if task not in results:
                        break
                    if self._is_failure(results[task]):
                        return task.get_name(), [(t.get_name(), results[t]) for t in self.tasks if t in results]
            return None, [(t.get_name(), results[t]) for t in self.tasks if t in results]
        finally:
            await self._cleanup()
