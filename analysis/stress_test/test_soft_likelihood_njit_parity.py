"""parity test: soft_likelihood_aggregate vs soft_likelihood_aggregate_njit.

asserts (best, gap) match exactly across:
  - 50+ random obs_lists of varied size and base mix
  - engineered small-gap cases (gap < 0.1) where the second-best
    computation is most likely to diverge
  - real obs_lists harvested from the decoder running on a 2k FASTQ subset
    (skipped if the FASTQ isn't available locally)

run with:  python analysis/stress_test/test_soft_likelihood_njit_parity.py
"""

import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from decoder import (
    BASES,
    BASE_TO_IDX,
    DESIGNED_RLS,
    MAX_OBS_RL,
    pack_log_channel,
    pack_obs_list,
    soft_likelihood_aggregate,
    soft_likelihood_aggregate_njit,
)


def make_synthetic_log_channel(rng, noise_sd=0.0):
    """build a dict-of-dict-of-array log_channel approximating the real one
    (peaked at the designed length, decays to either side). noise_sd=0 by
    default so engineered tie cases produce exactly tied scores."""
    log_channel = {}
    for b in BASES:
        log_channel[b] = {}
        for d in DESIGNED_RLS:
            arr = np.zeros(MAX_OBS_RL + 1, dtype=np.float64)
            for k in range(MAX_OBS_RL + 1):
                arr[k] = -((k - d) ** 2) / (2.0 * 1.5 ** 2)
                if noise_sd > 0:
                    arr[k] += rng.normal(0, noise_sd)
            log_channel[b][d] = arr
    return log_channel


def make_random_obs_list(rng, n):
    return [(int(rng.randint(1, MAX_OBS_RL + 1)), str(rng.choice(BASES)))
            for _ in range(n)]


def make_zerogap_obs_list(rng, n):
    """balanced split between k=5 and k=6 (the boundary between designed
    levels 4 and 7) -> log-likelihood scores for those two designs are
    exactly tied -> gap = 0. forces the second-best path to choose the
    matching tied score, which is the off-by-one trap."""
    if n % 2 != 0:
        n += 1
    base = str(rng.choice(BASES))
    obs = [(5, base)] * (n // 2) + [(6, base)] * (n // 2)
    return obs


def run_parity(obs_lists, log_channel, log_channel_arr):
    n_total = len(obs_lists)
    n_smallgap = 0
    n_fail = 0
    for i, obs_list in enumerate(obs_lists):
        best_ref, gap_ref = soft_likelihood_aggregate(obs_list, log_channel)
        obs_rls, bases_arr = pack_obs_list(obs_list)
        best_njit, gap_njit = soft_likelihood_aggregate_njit(
            obs_rls, bases_arr, log_channel_arr
        )
        if best_ref != best_njit:
            print(f"  obs_list[{i}] (n={len(obs_list)}): best mismatch "
                  f"(ref={best_ref}, njit={best_njit})")
            n_fail += 1
            continue
        # gap must match exactly (np.allclose rtol=0 atol=0)
        if not np.allclose(gap_ref, gap_njit, rtol=0, atol=0):
            print(f"  obs_list[{i}] (n={len(obs_list)}): gap mismatch "
                  f"(ref={gap_ref:.10f}, njit={gap_njit:.10f}, "
                  f"diff={abs(gap_ref - gap_njit):.3e})")
            n_fail += 1
            continue
        if gap_ref < 0.1:
            n_smallgap += 1
    return n_total, n_smallgap, n_fail


def test_parity_synthetic():
    rng = np.random.RandomState(42)
    log_channel = make_synthetic_log_channel(rng)
    log_channel_arr = pack_log_channel(log_channel)

    obs_lists = []
    # 30 random obs_lists with varied sizes
    for _ in range(30):
        n = int(rng.randint(1, 100))
        obs_lists.append(make_random_obs_list(rng, n))
    # 15 engineered zero-gap (perfect-tie) cases
    for _ in range(15):
        n = int(rng.randint(2, 30))
        obs_lists.append(make_zerogap_obs_list(rng, n))
    # 10 single-observation edge cases
    for _ in range(10):
        obs_lists.append(make_random_obs_list(rng, 1))

    n_total, n_smallgap, n_fail = run_parity(obs_lists, log_channel, log_channel_arr)

    if n_smallgap < 1:
        raise AssertionError(
            f"no small-gap (< 0.1) cases produced; engineering failed"
        )
    if n_fail:
        raise AssertionError(f"{n_fail}/{n_total} mismatches")

    print(f"  PASS: {n_total}/{n_total} synthetic obs_lists match exactly "
          f"({n_smallgap} with gap < 0.1)")


def test_parity_real_reads(fastq_path: str):
    """harvest real obs_lists by running parts of the decoder pipeline on
    a small FASTQ, then run the parity check on >= 50 of them."""
    from collections import defaultdict
    from decoder import (
        DATA_DIR,
        N_FRAMES,
        N_STRIPS,
        bits_to_pattern,
        build_empirical_channel,
        build_hmms,
        classify_address,
        extract_color_bounded,
        load_codebook,
        orient_and_trim,
        read_fastq,
    )

    codebook = load_codebook(os.path.join(REPO_ROOT, 'data', 'selected_codewords.tsv'))
    addr_patterns = {i: bits_to_pattern(codebook[i]) for i in range(len(codebook))}
    addr_hmms = build_hmms(codebook)

    # designed_map mirrors what main() builds: (strip_idx, ri) -> int intensity
    gt_frames = [np.load(os.path.join(DATA_DIR, f'beaver_frame_{fi}.npy'))
                 for fi in range(N_FRAMES)]
    gt_all = np.stack(gt_frames)
    designed_map = {}
    for si in range(N_STRIPS):
        fi = si // 128
        local = si % 128
        row = local // 4
        strip_col = local % 4
        for ri in range(24):
            p = ri // 3
            ch = ri % 3
            col = strip_col * 8 + p
            designed_map[(si, ri)] = int(gt_all[fi, row, col, ch])

    # per-strip color_bases (cyclic +1 mod 3 from last address base)
    strip_cbases = {}
    for si in range(N_STRIPS):
        pat = addr_patterns[si]
        last = BASES.index(pat[-1])
        cb = []
        for _ in range(24):
            last = (last + 1) % 3
            cb.append(BASES[last])
        strip_cbases[si] = cb

    strip_color_obs = defaultdict(lambda: [[] for _ in range(24)])

    n_processed = 0
    for _, seq in read_fastq(fastq_path):
        n_processed += 1
        payload, _ = orient_and_trim(seq)
        if payload is None or not (100 <= len(payload) <= 400):
            continue
        strip_idx, _, addr_end = classify_address(payload, codebook, addr_hmms)
        if strip_idx < 0 or strip_idx >= N_STRIPS or addr_end <= 0:
            continue
        result = extract_color_bounded(payload, addr_end, strip_cbases[strip_idx])
        if result is None:
            continue
        for ri in range(24):
            rl, base = result[ri]
            if rl > 0 and base in BASES:
                strip_color_obs[strip_idx][ri].append((rl, base))

    if n_processed < 50:
        print(f"  SKIP: only {n_processed} reads in fastq; need >= 50")
        return

    log_channel = build_empirical_channel(strip_color_obs, designed_map, codebook)
    log_channel_arr = pack_log_channel(log_channel)

    obs_lists = []
    for si in strip_color_obs:
        for ri in range(24):
            ol = strip_color_obs[si][ri]
            if ol:
                obs_lists.append(ol)

    if len(obs_lists) < 50:
        print(f"  SKIP: only {len(obs_lists)} obs_lists harvested; need >= 50")
        return
    if len(obs_lists) > 200:
        obs_lists = obs_lists[:200]

    n_total, n_smallgap, n_fail = run_parity(obs_lists, log_channel, log_channel_arr)
    if n_fail:
        raise AssertionError(f"{n_fail}/{n_total} mismatches on real obs_lists")
    print(f"  PASS: {n_total}/{n_total} real obs_lists match exactly "
          f"(harvested from {n_processed} reads, {n_smallgap} with gap < 0.1)")


if __name__ == "__main__":
    print("test_parity_synthetic:")
    test_parity_synthetic()

    fastq = sys.argv[1] if len(sys.argv) > 1 else "/tmp/jacquard_2k.fastq.gz"
    if os.path.exists(fastq):
        print(f"test_parity_real_reads ({fastq}):")
        test_parity_real_reads(fastq)
    else:
        print(f"(skipping real-reads test; {fastq} not found)")

    print("\nALL PASS")
