"""Repair stream limits, stderr capture, and cleanup in the pinned Kimina server.

The upstream REPL starts a stderr pipe but reads a different temporary file.
Drain that pipe continuously, preserve fatal Lean diagnostics on EOF, and clean
up files/tasks even when the child exited before ``close``. Its stdout reader's
64 KiB line limit also rejects large JSON responses; raise that per-reader limit
to a bounded 16 MiB.
"""
from __future__ import annotations

import asyncio
from datetime import datetime
import importlib
import os
import runpy
import signal


STDOUT_LINE_LIMIT = 16 * 1024 * 1024


async def _drain_stderr(repl):
    proc = repl.proc
    if proc is None or proc.stderr is None:
        return
    while chunk := await proc.stderr.read(4096):
        if not repl.error_file.closed:
            # send() seeks within this same file; always append new diagnostics.
            repl.error_file.seek(0, os.SEEK_END)
            repl.error_file.write(chunk.decode("utf-8", errors="replace"))
            repl.error_file.flush()


async def _finish_stderr(repl, timeout=1.0):
    task = getattr(repl, "_dream_stderr_task", None)
    if task is not None:
        try:
            await asyncio.wait_for(asyncio.shield(task), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            pass


def install_compatibility(repl_module=None):
    """Install narrowly scoped fixes on the pinned server's REPL class."""
    module = repl_module or importlib.import_module("server.repl")
    cls = module.Repl
    if getattr(cls, "_dream_compat_installed", False):
        return
    original_start = cls.start
    original_read = cls._read_response
    original_cpu_monitor = cls._cpu_monitor
    original_mem_monitor = cls._mem_monitor

    async def start(self):
        await original_start(self)
        if self.proc is not None and self.proc.stdout is not None:
            # Pinned start() exposes no subprocess kwargs. Adjust only its owned
            # stdout StreamReader, keeping upstream newline framing and the
            # standard over-limit error; stderr and other readers keep defaults.
            self.proc.stdout._limit = STDOUT_LINE_LIMIT
        self._dream_stderr_task = asyncio.create_task(_drain_stderr(self))

    async def read_response(self):
        raw = await original_read(self)
        if raw:
            return raw
        # Empty stdout is commonly a child crash. Let the child watcher and
        # stderr reader finish before reporting the actual failure to the client.
        proc = self.proc
        if proc is not None and proc.returncode is None:
            try:
                await asyncio.wait_for(proc.wait(), 1.0)
            except asyncio.TimeoutError:
                pass
        await _finish_stderr(self)
        stderr = ""
        if not self.error_file.closed:
            self.error_file.seek(0)
            stderr = self.error_file.read().strip()
        returncode = proc.returncode if proc is not None else None
        diagnostic = f"Lean REPL returned an empty response (exit status {returncode})"
        if stderr:
            diagnostic += ": " + stderr[-8192:]
        module.logger.error("{}", diagnostic)
        raise module.ReplError(diagnostic)

    async def cpu_monitor(self):
        try:
            await original_cpu_monitor(self)
        except (module.psutil.NoSuchProcess, ProcessLookupError):
            # Child disappearance is reported by the request's EOF handler.
            return

    async def mem_monitor(self):
        try:
            await original_mem_monitor(self)
        except (module.psutil.NoSuchProcess, ProcessLookupError):
            return

    async def close(self):
        self.last_check_at = datetime.now()
        proc = self.proc
        try:
            if proc is not None:
                if proc.stdin is not None:
                    proc.stdin.close()
                # start() creates a fresh process group whose ID is the child
                # PID. killpg(PID) also cleans surviving grandchildren after
                # the group leader exits; getpgid(dead_PID) raises too early.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                await proc.wait()
                if proc.stdin is not None:
                    try:
                        await proc.stdin.wait_closed()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                await _finish_stderr(self)
            if module.db.connected:
                await module.prisma.repl.update(
                    where={"uuid": str(self.uuid)},
                    data={"status": module.ReplStatus.STOPPED},
                )
        finally:
            tasks = [getattr(self, name, None) for name in
                     ("_cpu_task", "_mem_task", "_dream_stderr_task")]
            for task in tasks:
                if task is not None and not task.done():
                    task.cancel()
            await asyncio.gather(*(task for task in tasks if task is not None), return_exceptions=True)
            if not self.error_file.closed:
                self.error_file.close()

    cls.start = start
    cls._read_response = read_response
    cls._cpu_monitor = cpu_monitor
    cls._mem_monitor = mem_monitor
    cls.close = close
    cls._dream_compat_installed = True
    if repl_module is None:
        install_manager_cleanup(importlib.import_module("server.manager"), module)


def install_manager_cleanup(manager_module, repl_module):
    """Await shutdown closes that upstream only schedules in the background."""
    cls = manager_module.Manager
    if getattr(cls, "_dream_cleanup_installed", False):
        return

    async def cleanup(self):
        self._ensure_lock()
        async with self._cond:
            repls = set(self._free) | self._busy
            self._free.clear()
            self._busy.clear()
        closes = [asyncio.create_task(repl_module.close_verbose(repl)) for repl in repls]
        # Retirement/recycling may already have removed a REPL from both sets.
        # Its scheduled close still needs to finish before the event loop stops.
        close_code = repl_module.close_verbose.__code__
        pending = {task for task in asyncio.all_tasks()
                   if task is not asyncio.current_task()
                   and getattr(task.get_coro(), "cr_code", None) is close_code}
        outcomes = await asyncio.gather(*set(closes) | pending, return_exceptions=True)
        for outcome in outcomes:
            if isinstance(outcome, BaseException):
                manager_module.logger.error("REPL shutdown cleanup failed: {}", str(outcome))
        manager_module.logger.info("REPL manager cleanup completed")

    cls.cleanup = cleanup
    cls._dream_cleanup_installed = True


def main():
    install_compatibility()
    runpy.run_module("server", run_name="__main__")


if __name__ == "__main__":
    main()
