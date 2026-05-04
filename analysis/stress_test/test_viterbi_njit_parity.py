"""parity test: _run_viterbi (list-based) vs _run_viterbi_njit (CSR + njit).

asserts bit-identical (v, bp) on:
  1. 20+ simulated reads through a real ProfileHMM
  2. a synthetic 4-state HMM constructed with deliberate ties so the
     lowest-i tie-break is exercised on every step

run with:  python analysis/stress_test/test_viterbi_njit_parity.py
"""

import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from profile_hmm import (
    BASE_TO_IDX,
    NEG_INF_SENTINEL,
    ProfileHMM,
    _build_sparse,
    _build_sparse_csr,
    _run_viterbi,
    _run_viterbi_njit,
)


BASES_LIST = ['A', 'C', 'G', 'T']


def _normalize_neg(v):
    """map both -inf and the sentinel to a single value so the two
    representations compare equal element-wise."""
    return np.where((v == NEG_INF_SENTINEL) | np.isneginf(v), -np.inf, v)


def _run_both(hmm, seq):
    seq_idx = np.array([BASE_TO_IDX[b] for b in seq.upper()], dtype=np.int32)
    v_list, bp_list = _run_viterbi(
        seq_idx, hmm.n_states, hmm.predecessors, hmm.successors,
        hmm.log_emit, hmm.emitting_mask, hmm.start_idx,
    )
    v_njit, bp_njit = _run_viterbi_njit(
        seq_idx, hmm.n_states,
        hmm.pred_indptr, hmm.pred_indices, hmm.pred_log_trans,
        hmm.succ_indptr, hmm.succ_indices, hmm.succ_log_trans,
        hmm.log_emit_sentinel, hmm.emitting_mask_arr, hmm.start_idx,
    )
    return v_list, bp_list, v_njit, bp_njit


def test_parity_real_hmm():
    """20 simulated reads through a CATCAT pattern HMM."""
    pattern = "CATCAT"
    hmm = ProfileHMM(pattern)

    rng = np.random.RandomState(42)
    reads = []
    for _ in range(20):
        seq = ""
        for b in pattern:
            n = max(1, int(rng.poisson(3)))
            seq += b * n
            if rng.random() < 0.1:
                seq += rng.choice(BASES_LIST)
        seq += "G" * max(1, int(rng.poisson(4)))
        seq_chars = list(seq)
        for k in range(len(seq_chars)):
            if rng.random() < 0.02:
                seq_chars[k] = rng.choice(BASES_LIST)
        reads.append("".join(seq_chars))

    n_fail = 0
    for i, r in enumerate(reads):
        v_list, bp_list, v_njit, bp_njit = _run_both(hmm, r)

        if not np.array_equal(bp_list, bp_njit):
            print(f"  read {i} (len={len(r)}): bp MISMATCH")
            n_fail += 1
            continue

        v_a = _normalize_neg(v_list)
        v_b = _normalize_neg(v_njit)
        if not np.allclose(v_a, v_b, rtol=0, atol=0, equal_nan=False):
            mask = np.isfinite(v_a) & np.isfinite(v_b)
            max_diff = float(np.max(np.abs(v_a[mask] - v_b[mask]))) if mask.any() else 0.0
            print(f"  read {i} (len={len(r)}): v MISMATCH (max abs diff {max_diff:.3e})")
            n_fail += 1

    if n_fail:
        raise AssertionError(f"{n_fail}/20 reads diverged")
    print(f"  PASS: 20/20 reads bit-identical (v and bp)")


def test_parity_forced_tie():
    """4-state HMM with two predecessors of identical log_trans and matched
    incoming scores at the convergence state. exercises strict-> tie-break."""
    n_states = 4
    log_trans = np.full((n_states, n_states), -np.inf)
    log_trans[0, 1] = math.log(0.5)
    log_trans[0, 2] = math.log(0.5)
    log_trans[1, 3] = math.log(0.5)
    log_trans[2, 3] = math.log(0.5)

    log_emit = np.full((n_states, 4), -np.inf)
    log_emit[1, BASE_TO_IDX['A']] = math.log(1.0)
    log_emit[2, BASE_TO_IDX['A']] = math.log(1.0)
    log_emit[3, BASE_TO_IDX['A']] = math.log(1.0)

    emitting_mask = [False, True, True, True]

    (pred_indptr, pred_indices, pred_log_trans,
     succ_indptr, succ_indices, succ_log_trans) = _build_sparse_csr(log_trans)
    predecessors, successors = _build_sparse(log_trans)

    log_emit_sent = np.where(
        np.isneginf(log_emit), NEG_INF_SENTINEL, log_emit
    ).astype(np.float64)
    mask_arr = np.array(emitting_mask, dtype=np.bool_)

    seq_idx = np.array([BASE_TO_IDX['A'], BASE_TO_IDX['A']], dtype=np.int32)

    v_list, bp_list = _run_viterbi(
        seq_idx, n_states, predecessors, successors,
        log_emit, emitting_mask, 0,
    )
    v_njit, bp_njit = _run_viterbi_njit(
        seq_idx, n_states,
        pred_indptr, pred_indices, pred_log_trans,
        succ_indptr, succ_indices, succ_log_trans,
        log_emit_sent, mask_arr, 0,
    )

    # state 3 at t=2 has two equally-good predecessors (1 and 2);
    # both implementations must pick state 1 (lowest index).
    assert bp_list[2, 3] == 1, f"list bp[2,3] = {bp_list[2, 3]}, expected 1"
    assert bp_njit[2, 3] == 1, f"njit bp[2,3] = {bp_njit[2, 3]}, expected 1"

    if not np.array_equal(bp_list, bp_njit):
        raise AssertionError(f"bp diverged:\n  list:\n{bp_list}\n  njit:\n{bp_njit}")
    v_a = _normalize_neg(v_list)
    v_b = _normalize_neg(v_njit)
    if not np.allclose(v_a, v_b, rtol=0, atol=0, equal_nan=False):
        raise AssertionError(f"v diverged:\n  list:\n{v_a}\n  njit:\n{v_b}")
    print("  PASS: tie resolves to lowest-i predecessor; v + bp match")


if __name__ == "__main__":
    print("test_parity_real_hmm:")
    test_parity_real_hmm()
    print("test_parity_forced_tie:")
    test_parity_forced_tie()
    print("\nALL PASS")
