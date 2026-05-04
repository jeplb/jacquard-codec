"""parity test: bootstrap_accuracy_vs_depth vs the original inline loop.

the optimized bootstrap hoists full-coverage cells (n_obs <= depth, constant
across replicates) out of the replicate loop and reuses the all-reads
consensus in `recovered`, while leaving the subsampled cells -- and their
rng.choice draws -- in the exact same (strip, ri) order. this asserts the
returned depth_results are bit-identical to a faithful copy of the pre-refactor
loop on synthetic coverage that exercises both branches across all depths.

run with:  python analysis/stress_test/test_bootstrap_depth_parity.py
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.insert(0, REPO_ROOT)

from decoder import (
    BASES,
    MAX_OBS_RL,
    N_FRAMES,
    bootstrap_accuracy_vs_depth,
    pack_obs_list,
    soft_likelihood_aggregate_njit,
)


def _bootstrap_reference(packed_strip_obs, log_channel_arr, gt_all,
                         total_pixels, depths, n_reps, seed):
    """faithful copy of the original inline accuracy-vs-depth loop."""
    rng = np.random.RandomState(seed)
    depth_results = {}
    for depth in depths:
        accs = []
        for rep in range(n_reps):
            rec_sub = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
            for strip_idx, slot in packed_strip_obs.items():
                frame_i = strip_idx // 128
                local = strip_idx % 128
                row = local // 4
                strip_col = local % 4
                for ri, (obs_rls, bases_arr) in slot.items():
                    p_idx = ri // 3
                    ch = ri % 3
                    n_obs = obs_rls.shape[0]
                    if n_obs <= depth:
                        sample_rls = obs_rls
                        sample_bases = bases_arr
                    else:
                        idxs = rng.choice(n_obs, depth, replace=False)
                        sample_rls = obs_rls[idxs]
                        sample_bases = bases_arr[idxs]
                    pred, _ = soft_likelihood_aggregate_njit(
                        sample_rls, sample_bases, log_channel_arr
                    )
                    col = strip_col * 8 + p_idx
                    rec_sub[frame_i, row, col, ch] = pred
            accs.append(np.sum(gt_all == rec_sub) / total_pixels)
        accs = np.array(accs)
        lo, hi = np.percentile(accs, [2.5, 97.5])
        depth_results[depth] = (accs.mean(), lo, hi)
    return depth_results


def _consensus_recovered(packed_strip_obs, log_channel_arr):
    """all-reads soft-LL consensus, exactly as decoder.main() builds it."""
    recovered = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
    for strip_idx, slot in packed_strip_obs.items():
        frame_i = strip_idx // 128
        local = strip_idx % 128
        row = local // 4
        strip_col = local % 4
        for ri, (obs_rls, bases_arr) in slot.items():
            p_idx = ri // 3
            ch = ri % 3
            pred, _ = soft_likelihood_aggregate_njit(
                obs_rls, bases_arr, log_channel_arr
            )
            col = strip_col * 8 + p_idx
            recovered[frame_i, row, col, ch] = pred
    return recovered


def _make_synthetic(seed=7):
    rng = np.random.RandomState(seed)
    n_bases = 4
    n_designed = 4
    # random but fixed log-channel; magnitudes resemble the real (log-prob) one
    log_channel_arr = -np.abs(rng.normal(
        0, 3, size=(n_bases, n_designed, MAX_OBS_RL + 1))).astype(np.float64)

    # coverage values chosen to straddle every tested depth so both the
    # full-coverage and the subsample branch fire at each depth.
    coverage_choices = [1, 2, 4, 8, 12, 18, 26, 40]
    packed = {}
    # spread strips across frames/rows/cols
    strip_indices = list(range(0, 640, 9))[:60]
    for strip_idx in strip_indices:
        slot = {}
        for ri in range(24):
            if rng.random() < 0.15:
                continue  # leave some cells empty
            n = int(rng.choice(coverage_choices))
            obs_list = [(int(rng.randint(1, MAX_OBS_RL + 3)),
                         str(rng.choice(BASES))) for _ in range(n)]
            slot[ri] = pack_obs_list(obs_list)
        if slot:
            packed[strip_idx] = slot

    gt_all = rng.randint(0, 4, size=(N_FRAMES, 32, 32, 3)).astype(np.uint8)
    total_pixels = N_FRAMES * 32 * 32 * 3
    return packed, log_channel_arr, gt_all, total_pixels


def test_bootstrap_parity():
    packed, log_channel_arr, gt_all, total_pixels = _make_synthetic()
    recovered = _consensus_recovered(packed, log_channel_arr)
    depths = [1, 3, 5, 10, 15, 20, 25]

    ref = _bootstrap_reference(
        packed, log_channel_arr, gt_all, total_pixels, depths, 100, 42)

    # the optimized function prints per-depth lines; that's fine for the test
    opt = bootstrap_accuracy_vs_depth(
        packed, log_channel_arr, gt_all, recovered, total_pixels,
        depths, n_reps=100, seed=42)

    n_subsample_seen = 0
    for d in depths:
        rm, rlo, rhi = ref[d]
        om, olo, ohi = opt[d]
        assert rm == om, f"depth {d}: mean {rm!r} != {om!r}"
        assert rlo == olo, f"depth {d}: lo {rlo!r} != {olo!r}"
        assert rhi == ohi, f"depth {d}: hi {rhi!r} != {ohi!r}"
        if rlo != rhi:
            n_subsample_seen += 1

    assert n_subsample_seen >= 1, "no depth produced subsampling variance"
    print(f"  PASS: {len(depths)}/{len(depths)} depths bit-identical "
          f"({n_subsample_seen} with nonzero CI width)")


if __name__ == "__main__":
    print("test_bootstrap_parity:")
    test_bootstrap_parity()
    print("\nALL PASS")
