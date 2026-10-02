"""Deterministic 4c policy tests: injected monotonic clock, no sleeps/network."""
import unittest

from gateway.config import Settings
from gateway.limits import AdmissionController, Permit, Rejection


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.limits = AdmissionController(["alice", "bob"], 2, 3, 2, clock=self.clock)

    def acquire(self, caller="alice", inference=False):
        return self.limits.try_acquire(caller, inference=inference)

    def test_initial_burst_and_retry_after(self):
        for _ in range(3):
            permit = self.acquire()
            self.assertIsInstance(permit, Permit)
            permit.release()
        rejection = self.acquire()
        self.assertEqual(rejection, Rejection("rate_limit_exceeded", 1))
        self.assertEqual(self.limits.in_flight, 0)

    def test_fractional_refill_and_rejected_requests_do_not_consume_tokens(self):
        for _ in range(3):
            self.acquire().release()
        self.clock.advance(0.25)  # +0.5 tokens, not enough for a request.
        self.assertIsInstance(self.acquire(), Rejection)
        self.assertIsInstance(self.acquire(), Rejection)
        self.clock.advance(0.25)  # Another +0.5, now exactly one token.
        self.assertIsInstance(self.acquire(), Permit)
        self.assertIsInstance(self.acquire(), Rejection)

    def test_idle_refill_is_capped_at_burst(self):
        self.clock.advance(1000)
        for _ in range(3):
            self.assertIsInstance(self.acquire(), Permit)
        self.assertIsInstance(self.acquire(), Rejection)

    def test_callers_have_independent_buckets_and_unknown_ids_cannot_allocate(self):
        for _ in range(3):
            self.acquire("alice").release()
        self.assertIsInstance(self.acquire("alice"), Rejection)
        self.assertIsInstance(self.acquire("bob"), Permit)
        with self.assertRaises(KeyError):
            self.acquire("unregistered")
        self.assertEqual(set(self.limits._buckets), {"alice", "bob"})

    def test_global_cap_and_idempotent_release(self):
        first = self.acquire("alice", inference=True)
        second = self.acquire("bob", inference=True)
        self.assertEqual(self.limits.in_flight, 2)
        self.assertEqual(self.acquire("alice", inference=True), Rejection("gateway_busy", 1))
        self.assertEqual(self.limits.in_flight, 2)
        first.release()
        first.release()
        self.assertEqual(self.limits.in_flight, 1)
        third = self.acquire("bob", inference=True)
        self.assertIsInstance(third, Permit)
        self.assertEqual(self.limits.in_flight, 2)
        second.release()
        third.release()
        third.release()
        self.assertEqual(self.limits.in_flight, 0)

    def test_models_do_not_take_inference_slots_but_share_frequency_quota(self):
        first = self.acquire("alice", inference=True)
        second = self.acquire("bob", inference=True)
        model_request = self.acquire("alice", inference=False)
        self.assertIsInstance(model_request, Permit)
        self.assertEqual(self.limits.in_flight, 2)
        model_request.release()
        self.assertEqual(self.limits.in_flight, 2)
        self.assertEqual(self.acquire("alice", inference=True), Rejection("gateway_busy", 1))
        self.assertEqual(self.acquire("alice", inference=False), Rejection("rate_limit_exceeded", 1))
        first.release()
        second.release()

    def test_busy_attempts_spend_frequency_tokens_and_slot_release_does_not_refund(self):
        limits = AdmissionController(["alice", "bob"], 1, 1, 1, clock=self.clock)
        permit = limits.try_acquire("alice", inference=True)
        self.assertEqual(limits.try_acquire("bob", inference=True), Rejection("gateway_busy", 1))
        permit.release()
        self.assertEqual(limits.try_acquire("bob", inference=True), Rejection("rate_limit_exceeded", 1))
        self.assertEqual(limits.try_acquire("alice", inference=True), Rejection("rate_limit_exceeded", 1))
        self.clock.advance(1)
        self.assertIsInstance(limits.try_acquire("bob", inference=True), Permit)

    def test_backward_clock_does_not_create_tokens(self):
        for _ in range(3):
            self.acquire().release()
        self.clock.advance(-5)
        self.assertIsInstance(self.acquire(), Rejection)
        self.clock.advance(5)
        self.assertIsInstance(self.acquire(), Rejection)
        self.clock.advance(0.5)
        self.assertIsInstance(self.acquire(), Permit)

    def test_retry_after_rounds_up_to_whole_seconds(self):
        limits = AdmissionController(["alice"], 0.25, 1, 1, clock=self.clock)
        limits.try_acquire("alice", inference=False).release()
        self.assertEqual(limits.try_acquire("alice", inference=False).retry_after, 4)
        self.clock.advance(1.5)
        self.assertEqual(limits.try_acquire("alice", inference=False).retry_after, 3)
        self.clock.advance(2.5)
        self.assertIsInstance(limits.try_acquire("alice", inference=False), Permit)


class LimitConfigTests(unittest.TestCase):
    def test_invalid_limits(self):
        cases = [
            {"rate_limit_rps": 0}, {"rate_limit_rps": -1}, {"rate_limit_rps": True},
            {"rate_limit_rps": float("nan")}, {"rate_limit_rps": float("inf")},
            {"rate_limit_rps": 5e-324}, {"rate_limit_burst": 0}, {"rate_limit_burst": -1},
            {"rate_limit_burst": 1.5}, {"rate_limit_burst": True},
            {"max_in_flight": 0}, {"max_in_flight": -1}, {"max_in_flight": 1.5},
            {"max_in_flight": True}, {"max_in_flight": 101},
        ]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                Settings(api_keys={"test": "test-key"}, **case)


if __name__ == "__main__":
    unittest.main()
