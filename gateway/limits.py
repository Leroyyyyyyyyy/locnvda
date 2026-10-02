"""Single-event-loop, in-memory admission controls for stage 4c.

All mutations are synchronous (no await), so check/update is atomic within one
worker's event loop. Not shared across threads, workers, replicas or restarts.
"""
from dataclasses import dataclass
import math
import time


def validate_limits(rate, burst, max_in_flight):
    if (not isinstance(rate, (int, float)) or isinstance(rate, bool)
            or not math.isfinite(rate) or rate <= 0 or not math.isfinite(1 / rate)):
        raise ValueError("rate_limit_rps must be finite and positive with a finite reciprocal")
    for name, value in (("rate_limit_burst", burst), ("max_in_flight", max_in_flight)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")


@dataclass
class Bucket:
    tokens: float
    updated_at: float


@dataclass(frozen=True)
class Rejection:
    code: str
    retry_after: int


class Permit:
    """An idempotently releasable inference slot (or a no-slot models permit)."""

    def __init__(self, controller, inference):
        self._controller = controller
        self._inference = inference
        self._released = False

    def release(self):
        if not self._released:
            self._released = True
            if self._inference:
                self._controller._in_flight -= 1


class AdmissionController:
    def __init__(self, callers, rate, burst, max_in_flight, *, clock=None):
        validate_limits(rate, burst, max_in_flight)
        self._rate = rate
        self._burst = burst
        self._maximum = max_in_flight
        self._clock = clock if clock is not None else time.monotonic
        now = self._clock()
        # Allocate only for known caller IDs, never for arbitrary incoming keys.
        self._buckets = {caller: Bucket(float(burst), now) for caller in callers}
        self._in_flight = 0

    @property
    def in_flight(self):
        return self._in_flight

    def try_acquire(self, caller, *, inference):
        bucket = self._buckets[caller]
        now = self._clock()
        elapsed = max(0.0, now - bucket.updated_at)
        bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
        bucket.updated_at = max(bucket.updated_at, now)
        if bucket.tokens < 1:
            wait = max(1, math.ceil((1 - bucket.tokens) / self._rate))
            return Rejection("rate_limit_exceeded", wait)

        # Every authenticated attempt which passes the frequency check costs a
        # token, including busy/upstream-error attempts. No retry-token refunds.
        bucket.tokens -= 1
        if inference:
            if self._in_flight >= self._maximum:
                # A retry hint, not a promise that GPU capacity is free in 1 sec.
                return Rejection("gateway_busy", 1)
            self._in_flight += 1
        return Permit(self, inference)
