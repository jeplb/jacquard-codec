#!/usr/bin/env python3
"""bootstrap convergence: pixel accuracy vs read depth.

panel a: per-pixel-exact accuracy, soft-likelihood aggregation vs hard
         majority vote, both with 95% bootstrap CI bands.
panel b: per-rl-level accuracy under soft LL (RL = 3, 4, 7, 10).

uses real linear-amplicon nanopore reads. caches the slow pipeline
(primer trim + classify + bounded color extract + empirical channel)
to convergence_obs.pkl after the first run, so re-runs only redo the
bootstrap.
"""

import os
import sys
import time
import pickle
from collections import defaultdict

import numpy as np
import multiprocessing as mp

from numba import njit
from scipy.optimize import curve_fit

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, REPO_ROOT)
from tsv_records import write_record

from decoder import (
    BASES,
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
    DESIGNED_RLS,
    MAX_OBS_RL,
    MIN_PAYLOAD,
    MAX_PAYLOAD,
    pack_log_channel,
    pack_obs_list,
    soft_likelihood_aggregate_njit,
)

DATA_DIR = os.path.join(REPO_ROOT, 'data')

CACHE_PATH = os.path.join(SCRIPT_DIR, 'convergence_obs.pkl')
FIGURE_PATH = os.path.join(SCRIPT_DIR, 'convergence.pdf')
FIGURE_PNG_PATH = os.path.join(SCRIPT_DIR, 'convergence.png')
TSV_PATH = os.path.join(SCRIPT_DIR, 'convergence.tsv')

# depth grid: every integer 1-50 + extended tail (sampled with replacement
# beyond max-observed coverage to map out the asymptote firmly)
DEPTHS = list(range(1, 51)) + [60, 75, 100, 150, 200, 300]
N_REPS = 500

WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']
PHI = (1 + 5 ** 0.5) / 2


@njit(cache=True)
def hard_majority_vote_njit(obs_rls, bases, log_channel_arr):
    """per-obs ML classify, then majority vote with strict-> tie-break."""
    n_designed = log_channel_arr.shape[1]
    counts = np.zeros(n_designed, dtype=np.int64)
    for i in range(obs_rls.shape[0]):
        b = bases[i]
        k = obs_rls[i]
        best_di = 0
        best_val = log_channel_arr[b, 0, k]
        for di in range(1, n_designed):
            if log_channel_arr[b, di, k] > best_val:
                best_val = log_channel_arr[b, di, k]
                best_di = di
        counts[best_di] += 1
    best_idx = 0
    best_count = counts[0]
    for di in range(1, n_designed):
        if counts[di] > best_count:
            best_count = counts[di]
            best_idx = di
    return best_idx


# pipeline phase (cached)

def run_pipeline_phase(fastq_paths):
    """run primer trim + classify + bounded color extract + empirical channel.
    cached to disk; returns (strip_color_obs, log_channel, gt_all,
    designed_per_slot)."""
    if os.path.exists(CACHE_PATH):
        print(f"loading cached pipeline output from {CACHE_PATH}")
        with open(CACHE_PATH, 'rb') as f:
            return pickle.load(f)

    print("no cache; running pipeline (primer trim + classify + extract)...")

    codebook_path = os.path.join(DATA_DIR, 'selected_codewords.tsv')
    codebook = load_codebook(codebook_path)
    addr_patterns = {i: bits_to_pattern(codebook[i]) for i in range(len(codebook))}

    # per-strip color base sequence (cyclic +1 mod 3 from last address base)
    strip_cbases = {}
    for si in range(N_STRIPS):
        pat = addr_patterns[si]
        last = BASES.index(pat[-1])
        cb = []
        for _ in range(24):
            last = (last + 1) % 3
            cb.append(BASES[last])
        strip_cbases[si] = cb

    print("  building 640 extended HMMs...")
    t0 = time.time()
    addr_hmms = build_hmms(codebook)
    print(f"    done in {time.time()-t0:.1f}s")

    print("  primer trim + classify + bounded color extract...")
    t0 = time.time()
    strip_color_obs = defaultdict(lambda: [[] for _ in range(24)])
    n_total = 0
    n_classified = 0
    for fq in fastq_paths:
        for header, seq in read_fastq(fq):
            n_total += 1
            payload, _ = orient_and_trim(seq)
            if payload is None or not (MIN_PAYLOAD <= len(payload) <= MAX_PAYLOAD):
                continue
            strip_idx, method, addr_end = classify_address(payload, codebook, addr_hmms)
            if strip_idx < 0 or strip_idx >= N_STRIPS or addr_end <= 0:
                continue
            # only stage-1 reads contribute to color voting (matches paper)
            if method != 'stage1':
                continue
            result = extract_color_bounded(payload, addr_end, strip_cbases[strip_idx])
            if result is None:
                continue
            n_classified += 1
            for ri in range(24):
                rl, base = result[ri]
                if rl > 0 and base in BASES:
                    strip_color_obs[strip_idx][ri].append((rl, base))
            if n_total % 5000 == 0:
                print(f"    {n_total:,} reads scanned, {n_classified:,} stage-1 with color "
                      f"({time.time()-t0:.1f}s)")
    print(f"    {n_total:,} reads -> {n_classified:,} stage-1 with color "
          f"({time.time()-t0:.1f}s)")

    # ground truth + designed-per-slot map
    gt_frames = [np.load(os.path.join(DATA_DIR, f'beaver_frame_{i}.npy'))
                 for i in range(N_FRAMES)]
    gt_all = np.stack(gt_frames)

    designed_per_slot = {}
    for si in range(N_STRIPS):
        fi = si // 128
        local = si % 128
        row = local // 4
        strip_col = local % 4
        for ri in range(24):
            p = ri // 3
            ch = ri % 3
            col = strip_col * 8 + p
            designed_per_slot[(si, ri)] = int(gt_all[fi, row, col, ch])

    # build empirical channel
    print("  building empirical channel...")
    log_channel = build_empirical_channel(strip_color_obs, designed_per_slot, codebook)

    strip_color_obs = {si: list(slots) for si, slots in strip_color_obs.items()}

    out = {
        'strip_color_obs': strip_color_obs,
        'log_channel': log_channel,
        'gt_all': gt_all,
        'designed_per_slot': designed_per_slot,
    }
    print(f"  caching to {CACHE_PATH}")
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(out, f, protocol=4)
    return out


# bootstrap workers

_W = {}


def _pool_init(cache_path):
    """build packed structures once per worker."""
    global _W
    with open(cache_path, 'rb') as f:
        cache = pickle.load(f)
    strip_color_obs = cache['strip_color_obs']
    log_channel = cache['log_channel']
    gt_all = cache['gt_all']

    log_channel_arr = pack_log_channel(log_channel)
    packed = []  # list of (strip_idx, ri, obs_rls, bases, designed_idx)
    for strip_idx, slots in strip_color_obs.items():
        for ri in range(24):
            obs_list = slots[ri]
            if obs_list:
                obs_rls, bases_arr = pack_obs_list(obs_list)
                fi = strip_idx // 128
                local = strip_idx % 128
                row = local // 4
                strip_col = local % 4
                p = ri // 3
                ch = ri % 3
                col = strip_col * 8 + p
                designed_idx = int(gt_all[fi, row, col, ch])
                packed.append((strip_idx, ri, obs_rls, bases_arr, designed_idx))

    _W['packed'] = packed
    _W['log_channel_arr'] = log_channel_arr
    _W['gt_all'] = gt_all


def _depth_worker(args):
    """run N_REPS bootstrap replicates at one depth.
    returns (depth, accs_soft, accs_hard, accs_per_rl).
    """
    depth, n_reps, seed = args
    packed = _W['packed']
    log_channel_arr = _W['log_channel_arr']
    gt_all = _W['gt_all']
    rng = np.random.default_rng(seed)

    accs_soft = np.zeros(n_reps, dtype=np.float64)
    accs_hard = np.zeros(n_reps, dtype=np.float64)
    accs_per_rl = np.zeros((n_reps, 4), dtype=np.float64)

    n_slots = len(packed)
    slot_strip = np.zeros(n_slots, dtype=np.int64)
    slot_ri = np.zeros(n_slots, dtype=np.int64)
    slot_designed = np.zeros(n_slots, dtype=np.int64)
    for k, (si, ri, _, _, di) in enumerate(packed):
        slot_strip[k] = si
        slot_ri[k] = ri
        slot_designed[k] = di

    for rep in range(n_reps):
        rec_soft = np.full(gt_all.shape, -1, dtype=np.int64)
        rec_hard = np.full(gt_all.shape, -1, dtype=np.int64)

        for k in range(n_slots):
            strip_idx = slot_strip[k]
            ri = slot_ri[k]
            obs_rls = packed[k][2]
            bases_arr = packed[k][3]
            n = obs_rls.shape[0]

            # subsample WITH replacement
            idxs = rng.integers(0, n, size=depth)
            s_rls = obs_rls[idxs]
            s_bases = bases_arr[idxs]

            # decoder.soft_likelihood_aggregate_njit returns (best_idx, gap)
            pred_soft, _ = soft_likelihood_aggregate_njit(s_rls, s_bases, log_channel_arr)
            pred_hard = hard_majority_vote_njit(s_rls, s_bases, log_channel_arr)

            fi = strip_idx // 128
            local = strip_idx % 128
            row = local // 4
            strip_col = local % 4
            p = ri // 3
            ch = ri % 3
            col = strip_col * 8 + p
            rec_soft[fi, row, col, ch] = pred_soft
            rec_hard[fi, row, col, ch] = pred_hard

        # per-pixel-exact: all 3 CMY exact, restricted to covered pixels
        covered = np.all(rec_soft >= 0, axis=-1)
        gt_covered = gt_all[covered]
        soft_covered = rec_soft[covered].astype(np.int64)
        hard_covered = rec_hard[covered].astype(np.int64)
        soft_pixel_match = np.all(soft_covered == gt_covered, axis=-1)
        hard_pixel_match = np.all(hard_covered == gt_covered, axis=-1)
        n_cov = covered.sum()
        accs_soft[rep] = soft_pixel_match.sum() / max(n_cov, 1)
        accs_hard[rep] = hard_pixel_match.sum() / max(n_cov, 1)

        # per-rl-level (soft LL only)
        for designed_idx in range(4):
            mask = (slot_designed == designed_idx)
            if not mask.any():
                continue
            n_match = 0
            n_tot = 0
            for k in np.nonzero(mask)[0]:
                strip_idx = slot_strip[k]
                ri = slot_ri[k]
                fi = strip_idx // 128
                local = strip_idx % 128
                row = local // 4
                strip_col = local % 4
                p = ri // 3
                ch = ri % 3
                col = strip_col * 8 + p
                pred = rec_soft[fi, row, col, ch]
                if pred == designed_idx:
                    n_match += 1
                n_tot += 1
            accs_per_rl[rep, designed_idx] = n_match / max(n_tot, 1)

    return depth, accs_soft, accs_hard, accs_per_rl


# saturating fit

def saturation_model(d, acc_inf, acc_1, tau):
    return acc_inf - (acc_inf - acc_1) * np.exp(-d / tau)


def fit_saturation(depths, mean_accs):
    """fit acc(d) = acc_inf - (acc_inf - acc_1) * exp(-d/tau)."""
    depths = np.asarray(depths, dtype=np.float64)
    accs = np.asarray(mean_accs, dtype=np.float64)
    p0 = [accs.max(), accs.min(), 5.0]
    try:
        popt, _ = curve_fit(saturation_model, depths, accs, p0=p0, maxfev=10000)
    except Exception as e:
        print(f"    fit failed: {e}")
        return None
    acc_inf, acc_1, tau = popt

    def depth_to_frac(frac):
        if frac >= 1.0:
            return float('inf')
        return float(-tau * np.log(1 - frac))

    return {
        'acc_inf': float(acc_inf),
        'acc_1': float(acc_1),
        'tau': float(tau),
        'depth_to_90': depth_to_frac(0.90),
        'depth_to_99': depth_to_frac(0.99),
    }


# plotting

def plot_convergence(depths, soft_means, soft_lo, soft_hi,
                     hard_means, hard_lo, hard_hi,
                     per_rl_means, per_rl_lo, per_rl_hi,
                     fits):
    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['font.family'] = 'sans-serif'
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['font.size'] = 9
    import matplotlib.pyplot as plt

    fig_w = 7.0
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(fig_w, fig_w / PHI / 2 * 1.6)
    )

    depths = np.asarray(depths)

    # panel a: soft vs hard, per-pixel-exact
    ax_a.fill_between(depths, soft_lo, soft_hi, color=WONG[0], alpha=0.18)
    ax_a.plot(depths, soft_means, '-', color=WONG[0], linewidth=1.3,
              label='Soft likelihood', marker='o', markersize=2.5)
    ax_a.fill_between(depths, hard_lo, hard_hi, color=WONG[1], alpha=0.18)
    ax_a.plot(depths, hard_means, '-', color=WONG[1], linewidth=1.3,
              label='Hard majority vote', marker='s', markersize=2.5)
    fit_soft = fits.get('soft')
    if fit_soft is not None:
        ax_a.axhline(fit_soft['acc_inf'], color=WONG[0], ls=':', lw=0.7, alpha=0.6)
    fit_hard = fits.get('hard')
    if fit_hard is not None:
        ax_a.axhline(fit_hard['acc_inf'], color=WONG[1], ls=':', lw=0.7, alpha=0.6)
    ax_a.set_xscale('log')
    ax_a.set_xlabel('Read depth')
    ax_a.set_ylabel('Per-pixel exact accuracy')
    ax_a.legend(frameon=False, loc='lower right')
    for s in ('top', 'right'):
        ax_a.spines[s].set_visible(False)

    # panel b: soft LL, per-rl-level
    rl_labels = [f'RL = {rl}' for rl in DESIGNED_RLS]
    rl_colors = [WONG[0], WONG[2], WONG[3], WONG[4]]
    for i, (rl, label, color) in enumerate(zip(DESIGNED_RLS, rl_labels, rl_colors)):
        ax_b.fill_between(depths, per_rl_lo[:, i], per_rl_hi[:, i], color=color, alpha=0.18)
        ax_b.plot(depths, per_rl_means[:, i], '-', color=color, linewidth=1.3,
                  label=label, marker='o', markersize=2.5)
    ax_b.set_xscale('log')
    ax_b.set_xlabel('Read depth')
    ax_b.set_ylabel('Per-channel accuracy (soft LL)')
    ax_b.legend(frameon=False, loc='lower right')
    for s in ('top', 'right'):
        ax_b.spines[s].set_visible(False)

    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches='tight')
    fig.savefig(FIGURE_PNG_PATH, dpi=300, bbox_inches='tight')
    print(f"  saved {FIGURE_PATH}")
    print(f"  saved {FIGURE_PNG_PATH}")


# main

def main():
    fastq_paths = [p for p in sys.argv[1:] if os.path.exists(p)]
    if not fastq_paths and not os.path.exists(CACHE_PATH):
        print("usage: python convergence.py <fastq_path> [<fastq_path> ...]")
        print("(or place a prebuilt convergence_obs.pkl alongside this script)")
        sys.exit(1)

    cache = run_pipeline_phase(fastq_paths)
    n_slots_total = sum(
        sum(1 for slots in cache['strip_color_obs'][si]
            if slots) if si in cache['strip_color_obs'] else 0
        for si in range(N_STRIPS)
    )
    print(f"\n  {n_slots_total} populated (strip, position) slots")
    print(f"  bootstrap: {len(DEPTHS)} depths x {N_REPS} replicates "
          f"(with replacement)")

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"  pool: {n_workers} workers\n")

    base_seed = 42 * 1000000
    work = [(d, N_REPS, base_seed + d * 7919) for d in DEPTHS]

    t0 = time.time()
    results = {}
    with mp.Pool(processes=n_workers,
                 initializer=_pool_init,
                 initargs=(CACHE_PATH,)) as pool:
        for depth, accs_soft, accs_hard, accs_per_rl in pool.imap_unordered(
                _depth_worker, work):
            results[depth] = (accs_soft, accs_hard, accs_per_rl)
            elapsed = time.time() - t0
            print(f"  depth {depth:>3}: soft={accs_soft.mean():.4f} "
                  f"hard={accs_hard.mean():.4f} ({elapsed:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n  bootstrap complete in {elapsed:.1f}s\n")

    depths_sorted = sorted(results.keys())
    soft_means = np.array([results[d][0].mean() for d in depths_sorted])
    soft_lo = np.array([np.percentile(results[d][0], 2.5) for d in depths_sorted])
    soft_hi = np.array([np.percentile(results[d][0], 97.5) for d in depths_sorted])
    hard_means = np.array([results[d][1].mean() for d in depths_sorted])
    hard_lo = np.array([np.percentile(results[d][1], 2.5) for d in depths_sorted])
    hard_hi = np.array([np.percentile(results[d][1], 97.5) for d in depths_sorted])
    per_rl_means = np.array([results[d][2].mean(axis=0) for d in depths_sorted])
    per_rl_lo = np.array([np.percentile(results[d][2], 2.5, axis=0) for d in depths_sorted])
    per_rl_hi = np.array([np.percentile(results[d][2], 97.5, axis=0) for d in depths_sorted])

    print("  fitting saturation curves...")
    fits = {
        'soft': fit_saturation(depths_sorted, soft_means),
        'hard': fit_saturation(depths_sorted, hard_means),
    }
    for i, rl in enumerate(DESIGNED_RLS):
        fits[f'rl_{rl}'] = fit_saturation(depths_sorted, per_rl_means[:, i])

    for name, fit in fits.items():
        if fit is None:
            continue
        print(f"    {name}: acc_inf={fit['acc_inf']:.4f} tau={fit['tau']:.2f} "
              f"d90={fit['depth_to_90']:.1f} d99={fit['depth_to_99']:.1f}")

    record = {
        'depths': depths_sorted,
        'n_reps': N_REPS,
        'n_populated_slots': n_slots_total,
        'soft': {
            'mean': soft_means.tolist(),
            'ci_2.5': soft_lo.tolist(),
            'ci_97.5': soft_hi.tolist(),
        },
        'hard': {
            'mean': hard_means.tolist(),
            'ci_2.5': hard_lo.tolist(),
            'ci_97.5': hard_hi.tolist(),
        },
        'per_rl': {
            f'rl_{rl}': {
                'mean': per_rl_means[:, i].tolist(),
                'ci_2.5': per_rl_lo[:, i].tolist(),
                'ci_97.5': per_rl_hi[:, i].tolist(),
            }
            for i, rl in enumerate(DESIGNED_RLS)
        },
        'fits': fits,
    }
    write_record(TSV_PATH, record)
    print(f"  saved {TSV_PATH}")

    plot_convergence(
        depths_sorted, soft_means, soft_lo, soft_hi,
        hard_means, hard_lo, hard_hi,
        per_rl_means, per_rl_lo, per_rl_hi,
        fits,
    )


if __name__ == '__main__':
    main()
