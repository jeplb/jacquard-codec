"""parity test: HMMLibrary.viterbi_best (score-only ranking) vs scoring every
HMM with full viterbi.

the optimized viterbi_best ranks candidates with viterbi_score (no path /
no full-matrix sentinel conversion) and reconstructs the path only for the
winner. this asserts, on simulated reads across a multi-HMM library:
  - viterbi_score(seq) == viterbi(seq)[1] for every HMM (exact)
  - best_name, best_score, all_scores, and the winning path match the
    reference that scores every HMM with full viterbi

run with:  python analysis/stress_test/test_viterbi_best_parity.py
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
    HMMLibrary,
    ProfileHMM,
    ProfileHMMParameters,
)

BASES = ['A', 'C', 'T']
ALL_BASES = ['A', 'C', 'G', 'T']


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


def _reference_best(library, seq_idx):
    """old logic: full viterbi on every HMM, strict-greater argmax."""
    results = [hmm.viterbi(seq_idx) for hmm in library.hmms]
    best_idx = 0
    best_score = -math.inf
    all_scores = {}
    for i, (_, score) in enumerate(results):
        all_scores[library.names[i]] = score
        if score > best_score:
            best_score = score
            best_idx = i
    path, score = results[best_idx]
    return library.names[best_idx], path, score, all_scores


def _simulate(pat, rng):
    seq = ""
    for b in pat:
        n = max(1, int(rng.poisson(4)))
        seq += b * n
        if rng.random() < 0.08:
            seq += rng.choice(ALL_BASES)
    seq += "G" * max(1, int(rng.poisson(4)))
    chars = list(seq)
    for k in range(len(chars)):
        if rng.random() < 0.03:
            chars[k] = rng.choice(ALL_BASES)
    return "".join(chars)


def test_viterbi_best_parity():
    cb = load_codebook(os.path.join(REPO_ROOT, 'data', 'selected_codewords.tsv'))
    params = ProfileHMMParameters()
    params.hp_expected_length = 5.0

    # a 40-HMM library is enough to exercise ranking + tie-break paths
    idxs = list(range(0, 640, 16))[:40]
    library = HMMLibrary()
    for gi in idxs:
        library.add_hmm(str(gi), ProfileHMM(build_extended_pattern(cb[gi]), params))

    rng = np.random.RandomState(123)
    n_reads = 60
    n_fail = 0
    for r in range(n_reads):
        src = idxs[rng.randint(len(idxs))]
        read = _simulate(bits_to_pattern(cb[src]), rng)
        seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in read.upper()],
                           dtype=np.int32)

        # viterbi_score must equal viterbi()[1] exactly for every HMM
        for hmm in library.hmms:
            s_only = hmm.viterbi_score(seq_idx)
            _, s_full = hmm.viterbi(seq_idx)
            if not (s_only == s_full or (math.isinf(s_only) and math.isinf(s_full))):
                print(f"  read {r}: viterbi_score {s_only!r} != viterbi {s_full!r}")
                n_fail += 1

        ref_name, ref_path, ref_score, ref_scores = _reference_best(library, seq_idx)
        name, path, score, scores = library.viterbi_best(seq_idx)

        if name != ref_name:
            print(f"  read {r}: best_name {name} != {ref_name}")
            n_fail += 1
        if not (score == ref_score):
            print(f"  read {r}: best_score {score!r} != {ref_score!r}")
            n_fail += 1
        if path != ref_path:
            print(f"  read {r}: winning path differs")
            n_fail += 1
        for k in ref_scores:
            if scores[k] != ref_scores[k]:
                print(f"  read {r}: all_scores[{k}] {scores[k]!r} != {ref_scores[k]!r}")
                n_fail += 1
                break

    if n_fail:
        raise AssertionError(f"{n_fail} mismatches across {n_reads} reads")
    print(f"  PASS: {n_reads}/{n_reads} reads match "
          f"(scores, argmax, all_scores, winning path) across {len(idxs)} HMMs")


if __name__ == "__main__":
    print("test_viterbi_best_parity:")
    test_viterbi_best_parity()
    print("\nALL PASS")
