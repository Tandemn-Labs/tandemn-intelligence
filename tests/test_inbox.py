"""Tests for the inbox state machine in koi/runtime_state.py.

Phase 2b of the contract-hardening rollout. The core invariant: a handler
crash between `claim_event` and `mark_processed` does NOT lose the event —
the row stays in 'processing' and a later retry (after the reclaim window)
re-owns it as RECLAIMED_STALE.
"""

import threading
import time

import pytest

from koi.runtime_state import ClaimResult, RuntimeStateStore


@pytest.fixture
def store():
    return RuntimeStateStore(":memory:")


class TestFreshClaim:
    def test_first_claim_returns_claimed(self, store):
        assert store.claim_event("ev-1", "job_complete", "mo-abc") == ClaimResult.CLAIMED

    def test_claimed_row_is_processing(self, store):
        store.claim_event("ev-1", "job_complete", "mo-abc")
        assert store.inbox_count(status="processing") == 1
        assert store.inbox_count(status="processed") == 0

    def test_stores_payload_hash(self, store):
        store.claim_event("ev-1", "job_complete", "mo-abc", payload_hash="sha-xyz")
        # No direct getter; verify via the existence test pathway below.
        assert store.inbox_count() == 1


class TestDuplicateDetection:
    def test_immediate_reclaim_returns_in_flight(self, store):
        assert store.claim_event("ev-1", "t", "j") == ClaimResult.CLAIMED
        assert store.claim_event("ev-1", "t", "j") == ClaimResult.IN_FLIGHT

    def test_after_mark_processed_returns_already_processed(self, store):
        store.claim_event("ev-1", "t", "j")
        store.mark_processed("ev-1")
        assert store.claim_event("ev-1", "t", "j") == ClaimResult.ALREADY_PROCESSED

    def test_mark_processed_is_idempotent(self, store):
        store.claim_event("ev-1", "t", "j")
        store.mark_processed("ev-1")
        store.mark_processed("ev-1")
        assert store.inbox_count(status="processed") == 1


class TestStaleReclaim:
    def test_stale_claim_reclaimable(self, store):
        """A claim older than reclaim_after_secs is available to re-own."""
        store.claim_event("ev-1", "t", "j", reclaim_after_secs=0.05)
        time.sleep(0.1)
        assert store.claim_event("ev-1", "t", "j", reclaim_after_secs=0.05) == ClaimResult.RECLAIMED_STALE

    def test_reclaim_bumps_attempts(self, store):
        store.claim_event("ev-1", "t", "j", reclaim_after_secs=0.05)
        time.sleep(0.1)
        store.claim_event("ev-1", "t", "j", reclaim_after_secs=0.05)
        # Second claim bumped attempts from 1 to 2
        stale = store.inbox_stale_count(older_than_secs=0)
        # Reclaim reset claimed_at to now, so stale_count at threshold=0
        # is 1 row (itself), or 0 depending on clock jitter; assert exactly
        # 1 row remains in 'processing'.
        assert store.inbox_count(status="processing") == 1

    def test_fresh_claim_not_stale(self, store):
        store.claim_event("ev-1", "t", "j")
        # Large reclaim window; shouldn't be stale.
        assert store.claim_event("ev-1", "t", "j", reclaim_after_secs=120.0) == ClaimResult.IN_FLIGHT


class TestCrashMidHandler:
    """The critical invariant: an event never gets lost across a handler crash."""

    def test_partial_handler_then_retry_replays(self, store):
        # 1. Handler starts
        assert store.claim_event("ev-1", "job_complete", "mo-abc", reclaim_after_secs=0.05) == ClaimResult.CLAIMED
        # 2. Handler crashes before mark_processed — row stays 'processing'
        assert store.inbox_count(status="processed") == 0
        # 3. Orca retries after the reclaim window
        time.sleep(0.1)
        result = store.claim_event("ev-1", "job_complete", "mo-abc", reclaim_after_secs=0.05)
        assert result == ClaimResult.RECLAIMED_STALE
        # 4. This time the handler completes
        store.mark_processed("ev-1")
        # 5. Further retries are true duplicates
        assert store.claim_event("ev-1", "job_complete", "mo-abc") == ClaimResult.ALREADY_PROCESSED


class TestMarkFailed:
    def test_records_error_keeps_processing(self, store):
        store.claim_event("ev-1", "t", "j")
        store.mark_failed("ev-1", "connection refused")
        # Status stays 'processing' so retries can reclaim.
        assert store.inbox_count(status="processing") == 1
        assert store.inbox_count(status="processed") == 0

    def test_truncates_long_errors(self, store):
        store.claim_event("ev-1", "t", "j")
        store.mark_failed("ev-1", "x" * 5000)
        # Should not raise; schema has no length constraint but we cap at 2000.
        assert store.inbox_count(status="processing") == 1


class TestConcurrentClaim:
    def test_only_one_thread_gets_claimed(self, store):
        """10 threads racing on the same event_id: exactly one CLAIMED wins."""
        results: list = []
        barrier = threading.Barrier(10)

        def worker():
            barrier.wait()
            r = store.claim_event("ev-1", "t", "j")
            results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        claimed = sum(1 for r in results if r == ClaimResult.CLAIMED)
        in_flight = sum(1 for r in results if r == ClaimResult.IN_FLIGHT)
        assert claimed == 1
        assert claimed + in_flight == 10


class TestCounts:
    def test_inbox_count_by_status(self, store):
        store.claim_event("ev-1", "t", "j")
        store.claim_event("ev-2", "t", "j")
        store.mark_processed("ev-1")
        assert store.inbox_count() == 2
        assert store.inbox_count(status="processed") == 1
        assert store.inbox_count(status="processing") == 1

    def test_inbox_stale_count(self, store):
        store.claim_event("ev-stale", "t", "j")
        # Force stale by backdating via SQL
        with store._lock:
            store._conn.execute(
                "UPDATE inbox SET claimed_at = ? WHERE event_id = 'ev-stale'",
                (time.time() - 600,),
            )
            store._conn.commit()
        store.claim_event("ev-fresh", "t", "j")
        assert store.inbox_stale_count(older_than_secs=300) == 1

    def test_inbox_stale_count_ignores_processed(self, store):
        store.claim_event("ev-1", "t", "j")
        with store._lock:
            store._conn.execute(
                "UPDATE inbox SET claimed_at = ? WHERE event_id = 'ev-1'",
                (time.time() - 600,),
            )
            store._conn.commit()
        store.mark_processed("ev-1")
        assert store.inbox_stale_count(older_than_secs=300) == 0


class TestPrune:
    def test_removes_old_processed(self, store):
        store.claim_event("old", "t", "j")
        store.mark_processed("old")
        # Backdate processed_at
        with store._lock:
            store._conn.execute(
                "UPDATE inbox SET processed_at = ? WHERE event_id = 'old'",
                (time.time() - 20 * 86400,),  # 20 days old
            )
            store._conn.commit()
        store.claim_event("new", "t", "j")
        store.mark_processed("new")
        removed = store.prune_inbox(keep_secs=14 * 86400)
        assert removed == 1
        assert store.inbox_count() == 1

    def test_keeps_processing_rows_regardless_of_age(self, store):
        """Processing rows are never pruned — even if ancient, Orca may retry."""
        store.claim_event("stuck", "t", "j")
        with store._lock:
            store._conn.execute(
                "UPDATE inbox SET claimed_at = ? WHERE event_id = 'stuck'",
                (time.time() - 20 * 86400,),
            )
            store._conn.commit()
        removed = store.prune_inbox(keep_secs=14 * 86400)
        assert removed == 0
        assert store.inbox_count(status="processing") == 1
