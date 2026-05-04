"""parity test: refactored HMM construction vs the original dense-scan path.

the optimized ProfileHMM builds predecessors / successors / CSR arrays from
an edge dict recorded during _build_transitions, skipping the two full
n-by-n dense scans the original used. this asserts the optimized object's
sparse structures are bit-identical to the reference built by scanning the
dense log_trans the same way the original code did.

covers every codebook pattern (640 extended HMMs) plus the demo CATCAT HMM.

run with:  python analysis/stress_test/test_build_sparse_parity.py
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from profile_hmm import (
    ProfileHMM,
    ProfileHMMParameters,
    _build_sparse,
)

BASES = ['A', 'C', 'T']


def _build_sparse_csr_ref(log_trans):
    """faithful copy of the ORIGINAL _build_sparse_csr (double dense scan).

    kept here so the test pins the optimized construction against the exact
    arrays the pre-refactor code produced, independent of how the module
    helper is now implemented.
    """
    import math
    n = log_trans.shape[0]
    pred_lists = [[] for _ in range(n)]
    succ_lists = [[] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            v = float(log_trans[i, j])
            if v > -math.inf:
                pred_lists[j].append((i, v))
                succ_lists[i].append((j, v))

    pred_indptr = np.zeros(n + 1, dtype=np.int64)
    succ_indptr = np.zeros(n + 1, dtype=np.int64)
    for k in range(n):
        pred_indptr[k + 1] = pred_indptr[k] + len(pred_lists[k])
        succ_indptr[k + 1] = succ_indptr[k] + len(succ_lists[k])

    pred_nnz = int(pred_indptr[-1])
    succ_nnz = int(succ_indptr[-1])
    pred_indices = np.zeros(pred_nnz, dtype=np.int64)
    pred_log_trans = np.zeros(pred_nnz, dtype=np.float64)
    succ_indices = np.zeros(succ_nnz, dtype=np.int64)
    succ_log_trans = np.zeros(succ_nnz, dtype=np.float64)
    for k in range(n):
        for off, (i, v) in enumerate(pred_lists[k]):
            pred_indices[pred_indptr[k] + off] = i
            pred_log_trans[pred_indptr[k] + off] = v
        for off, (j, v) in enumerate(succ_lists[k]):
            succ_indices[succ_indptr[k] + off] = j
            succ_log_trans[succ_indptr[k] + off] = v
    return (pred_indptr, pred_indices, pred_log_trans,
            succ_indptr, succ_indices, succ_log_trans)


def _assert_lists_equal(a, b, label):
    assert len(a) == len(b), f"{label}: length {len(a)} != {len(b)}"
    for k, (la, lb) in enumerate(zip(a, b)):
        assert la == lb, f"{label}[{k}]: {la} != {lb}"


def _check_one(pattern, params):
    hmm = ProfileHMM(pattern, params)

    # reference sparse structures from the dense matrix the optimized object
    # still materialises in self.log_trans
    ref_pred, ref_succ = _build_sparse(hmm.log_trans)
    (r_pi, r_pidx, r_pv, r_si, r_sidx, r_sv) = _build_sparse_csr_ref(hmm.log_trans)

    _assert_lists_equal(hmm.predecessors, ref_pred, "predecessors")
    _assert_lists_equal(hmm.successors, ref_succ, "successors")

    assert np.array_equal(hmm.pred_indptr, r_pi), "pred_indptr"
    assert np.array_equal(hmm.pred_indices, r_pidx), "pred_indices"
    assert np.array_equal(hmm.pred_log_trans, r_pv), "pred_log_trans"
    assert np.array_equal(hmm.succ_indptr, r_si), "succ_indptr"
    assert np.array_equal(hmm.succ_indices, r_sidx), "succ_indices"
    assert np.array_equal(hmm.succ_log_trans, r_sv), "succ_log_trans"


def bits_to_pattern(bits):
    cur = 0
    pat = []
    for b in bits:
        cur = (cur + 1) % 3 if b == 0 else (cur + 2) % 3
        pat.append(BASES[cur])
    return ''.join(pat)


def build_extended_pattern(bits):
    ap = bits_to_pattern(bits)
    last = BASES.index(ap[-1])
    cb = []
    for _ in range(24):
        last = (last + 1) % 3
        cb.append(BASES[last])
    return ap + ''.join(cb)


def load_codebook(path):
    cw = []
    with open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 3:
                cw.append(np.array([int(b) for b in p[1]], dtype=np.uint8))
    return np.array(cw)


def test_parity_codebook():
    cb = load_codebook(os.path.join(REPO_ROOT, 'data', 'selected_codewords.tsv'))
    params = ProfileHMMParameters()
    params.hp_expected_length = 5.0
    for i in range(len(cb)):
        _check_one(build_extended_pattern(cb[i]), params)
    print(f"  PASS: {len(cb)}/{len(cb)} extended HMMs bit-identical "
          f"(predecessors, successors, CSR)")


def test_parity_demo():
    _check_one("CATCAT", ProfileHMMParameters())
    print("  PASS: CATCAT demo HMM bit-identical")


if __name__ == "__main__":
    print("test_parity_demo:")
    test_parity_demo()
    print("test_parity_codebook:")
    test_parity_codebook()
    print("\nALL PASS")
