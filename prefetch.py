"""Overlap data generation with the network pass.

At pathology_prob=1.0 the two stages are close to balanced -- generation
averages ~0.35s over the cheap-repeat cycle, the batch-of-2 forward/backward
~0.38s -- and they run strictly in sequence, so each step costs their sum.
Producing the next case while the current one trains makes the step cost
max(gen, net) instead of gen + net.

This only pays off when the stages are comparable -- if generation were far
slower than the network pass, prefetching would buy little, since the step
cost would already be dominated by generation regardless of overlap.

A thread rather than a process. The generator already runs on the GPU, so a
forked worker would need its own CUDA context and the tensors would have to
cross a process boundary. Threads share the context and the tensors directly.
The GIL is not the bottleneck: the generator spends its time inside torch
kernels and numpy, both of which release it.

Ordering, contents and RNG draws are unchanged -- the same cases arrive in the
same sequence. Only the timing differs.

Usage:
    from prefetch import Prefetcher
    gen = Prefetcher(paired_pathology_generator(...), depth=2)
    case = next(gen)          # same interface
    ...
    gen.close()               # optional; also closes at interpreter exit
"""
import queue
import threading

import torch


class Prefetcher:
    """Wrap an iterator so the next item is produced on a background thread.

    depth=2 is deliberate: depth 1 leaves the producer idle whenever the
    consumer is mid-step, and deeper queues hold more GPU tensors alive
    without further benefit, since the consumer can only ever be one step
    ahead of what it needs.
    """

    def __init__(self, iterator, depth=2):
        self._it = iterator
        self._q = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._exc = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        # The wrapped generator yields from inside `with torch.no_grad():`.
        # Grad mode is thread-local in PyTorch, so this thread having it off
        # does not affect the training thread -- but set it explicitly rather
        # than relying on that, since the consumer's own set_grad_enabled(True)
        # is what actually protects it.
        try:
            while not self._stop.is_set():
                item = next(self._it)
                while not self._stop.is_set():
                    try:
                        self._q.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        continue
        except StopIteration:
            self._q.put(_SENTINEL)
        except BaseException as e:          # noqa: BLE001 -- re-raised in consumer
            self._exc = e
            try:
                self._q.put(_SENTINEL, timeout=1.0)
            except queue.Full:
                pass

    def __iter__(self):
        return self

    def __next__(self):
        item = self._q.get()
        if item is _SENTINEL:
            if self._exc is not None:
                # Surface producer failures in the consumer rather than losing
                # them on a daemon thread, where training would silently hang.
                raise self._exc
            raise StopIteration
        return item

    def close(self):
        self._stop.set()
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)


class _Sentinel:
    pass


_SENTINEL = _Sentinel()