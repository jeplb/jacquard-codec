#!/usr/bin/env python3
"""
diagnose and recalibrate the joint codec color decoder using real data.

extracts empirical channel from exact-match reads, fits new laplace
parameters, checks for position/level-dependent effects, and re-runs
the decoder with recalibrated parameters.
"""

import os, sys, gzip, time
import numpy as np
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from tsv_records import write_record

from profile_hmm import ProfileHMM, ProfileHMMParameters, BASE_TO_IDX, StateType

DATA_DIR = os.path.join(REPO_ROOT, 'data')
FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

FWD_PRIMER = "TGGCTCTATGTCGGGAAACCCCTAT"
REV_PRIMER = "ATGTGGCTCTGCCCATCGATTTGTC"
BASES = ['A', 'C', 'T']
ADDRESS_RL = 5
INTENSITY_TO_RL = {0: 3, 1: 4, 2: 7, 3: 10}
RL_TO_INTENSITY = {3: 0, 4: 1, 7: 2, 10: 3}
DESIGNED_RLS = [3, 4, 7, 10]
N_STRIPS = 640
N_FRAMES = 5
RUN_PARSE_TRIM = 120
VITERBI_TRIM = 280

COMP = str.maketrans('ACGT', 'TGCA')
def revcomp(seq): return seq.translate(COMP)[::-1]

FWD_CORE = FWD_PRIMER[5:20]
REV_CORE = REV_PRIMER[5:20]

# old params from babbage pool
OLD_LAPLACE = {'A': (0, 1.87), 'C': (-2, 1.43), 'T': (0, 1.51)}


def load_codebook(path):
    cw = []
    with open(path) as f:
        for line in f:
            p = line.strip().split('\t')
            if len(p) == 3:
                cw.append(np.array([int(b) for b in p[1]], dtype=np.uint8))
    return np.array(cw)


def bits_to_pattern(bits):
    current = 0; pat = []
    for b in bits:
        current = (current + 1) % 3 if b == 0 else (current + 2) % 3
        pat.append(BASES[current])
    return ''.join(pat)


def _trim_fwd_rev(seq):
    plen_f = len(FWD_PRIMER)
    head = seq[:60]
    pos_f = head.find(FWD_CORE)
    if pos_f < 0:
        return None
    payload_start = pos_f - 5 + plen_f
    if payload_start < 0:
        payload_start = pos_f + len(FWD_CORE)
    tail = seq[-60:]
    pos_r = tail.find(REV_CORE)
    if pos_r >= 0:
        payload_end = len(seq) - 60 + pos_r - 5
        if payload_end > payload_start:
            return seq[payload_start:payload_end]
    return seq[payload_start:]


def orient_and_trim(seq):
    payload = _trim_fwd_rev(seq)
    if payload is not None:
        return payload
    rc_rev_core = revcomp(REV_CORE)
    if seq[:60].find(rc_rev_core) >= 0:
        flipped = revcomp(seq)
        payload = _trim_fwd_rev(flipped)
        if payload is not None:
            return payload
    return None


def parse_hp_runs(seq, max_nt=120):
    trimmed = seq[:max_nt]
    runs = []
    for b in trimmed:
        if b not in 'ACT':
            continue
        if runs and runs[-1][0] == b:
            runs[-1] = (b, runs[-1][1] + 1)
        else:
            runs.append((b, 1))
    return runs


def merge_noise_runs(runs):
    if len(runs) <= 2:
        return runs
    changed = True
    while changed:
        changed = False
        merged = [runs[0]]
        i = 1
        while i < len(runs):
            if (i + 1 < len(runs) and
                    runs[i][1] == 1 and
                    merged[-1][0] == runs[i + 1][0]):
                merged[-1] = (merged[-1][0], merged[-1][1] + 1 + runs[i + 1][1])
                i += 2; changed = True
            else:
                merged.append(runs[i]); i += 1
        runs = merged
    return runs


def classify_address_exact(payload, codebook):
    """run parser only. returns (strip_idx, hamming_dist) or (-1, 99)."""
    runs = parse_hp_runs(payload, RUN_PARSE_TRIM)
    runs = merge_noise_runs(runs)
    if len(runs) < 16:
        return -1, 99
    BASE_MAP = {'A': 0, 'C': 1, 'T': 2}
    bits = []
    prev = 0
    for base, _ in runs[:16]:
        obs = BASE_MAP[base]
        advance = (obs - prev) % 3
        bits.append(0 if advance == 1 else 1)
        prev = obs
    bits_arr = np.array(bits, dtype=np.uint8)
    dists = np.sum(codebook != bits_arr, axis=1)
    best_idx = int(np.argmin(dists))
    return best_idx, int(dists[best_idx])


def extract_color_rls(payload, addr_pattern, hmm_cache):
    """returns list of 24 observed run lengths and the color bases."""
    last = BASES.index(addr_pattern[-1])
    color_bases = []
    for _ in range(24):
        last = (last + 1) % 3
        color_bases.append(BASES[last])
    full_pattern = addr_pattern + ''.join(color_bases)

    if full_pattern not in hmm_cache:
        params = ProfileHMMParameters()
        params.hp_expected_length = float(ADDRESS_RL)
        hmm_cache[full_pattern] = ProfileHMM(full_pattern, params)
    hmm = hmm_cache[full_pattern]

    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in payload[:VITERBI_TRIM].upper()],
                       dtype=np.int32)
    if len(seq_idx) < 10:
        return None, color_bases

    path, _ = hmm.viterbi(seq_idx)

    counts = {}
    for t, (st, pos, _) in enumerate(path):
        if st == StateType.MATCH and pos is not None:
            counts[pos] = counts.get(pos, 0) + 1

    return [counts.get(pos, 0) for pos in range(16, 40)], color_bases


def read_fastq(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header: break
            seq = f.readline().strip()
            f.readline(); f.readline()
            yield seq


def nearest_intensity(obs_rl, base, laplace_params):
    mu = laplace_params.get(base, (0, 1.0))[0]
    corrected = obs_rl - mu
    dists = [abs(corrected - d) for d in DESIGNED_RLS]
    return dists.index(min(dists))


def nearest_intensity_custom(obs_rl, base, laplace_params, designed_rls):
    mu = laplace_params.get(base, (0, 1.0))[0]
    corrected = obs_rl - mu
    dists = [abs(corrected - d) for d in designed_rls]
    return dists.index(min(dists))


def main():
    if len(sys.argv) < 2:
        print("usage: calibrate.py <fastq>")
        sys.exit(1)
    fastq = sys.argv[1]

    codebook = load_codebook(os.path.join(DATA_DIR, 'selected_codewords.tsv'))
    addr_patterns = {i: bits_to_pattern(codebook[i]) for i in range(len(codebook))}

    gt_frames = [np.load(os.path.join(DATA_DIR, f'beaver_frame_{i}.npy'))
                 for i in range(N_FRAMES)]
    gt_all = np.stack(gt_frames)

    # precompute designed intensities per strip per color position
    designed = {}  # (strip_idx, ri) -> designed_intensity
    for si in range(N_STRIPS):
        fi = si // 128
        local = si % 128
        row = local // 4
        strip_col = local % 4
        for p in range(8):
            for ch in range(3):
                ri = p * 3 + ch
                col = strip_col * 8 + p
                designed[(si, ri)] = int(gt_all[fi, row, col, ch])

    # precompute color bases per strip
    strip_color_bases = {}
    for si in range(N_STRIPS):
        pat = addr_patterns[si]
        last = BASES.index(pat[-1])
        cb = []
        for _ in range(24):
            last = (last + 1) % 3
            cb.append(BASES[last])
        strip_color_bases[si] = cb

    print("STEP 1: extract empirical channel from exact-match reads")
    t0 = time.time()

    # obs_by_base_level[base][designed_rl] = list of observed RLs
    obs_by_base_level = {b: {rl: [] for rl in DESIGNED_RLS} for b in BASES}
    # also track by position
    obs_by_position = [[] for _ in range(24)]  # list of (designed_rl, obs_rl)

    hmm_cache = {}
    n_exact = 0
    n_total = 0

    for seq in read_fastq(fastq):
        payload = orient_and_trim(seq)
        if payload is None or len(payload) < 100 or len(payload) > 400:
            continue
        n_total += 1

        si, hdist = classify_address_exact(payload, codebook)
        if hdist > 0 or si < 0:
            continue
        n_exact += 1

        color_rls, color_bases = extract_color_rls(payload, addr_patterns[si], hmm_cache)
        if color_rls is None:
            continue

        for ri in range(24):
            obs_rl = color_rls[ri]
            if obs_rl <= 0:
                continue
            des_int = designed[(si, ri)]
            des_rl = INTENSITY_TO_RL[des_int]
            base = color_bases[ri]
            obs_by_base_level[base][des_rl].append(obs_rl)
            obs_by_position[ri].append((des_rl, obs_rl))

    print(f"  total reads: {n_total:,}")
    print(f"  exact match (dist=0): {n_exact:,}")
    print(f"  time: {time.time()-t0:.1f}s")

    # step 2: fit new Laplace parameters
    print("\nSTEP 2: fit new laplace parameters")
    new_laplace = {}

    print(f"\n  {'base':>4} {'des_rl':>6} {'N':>7} {'old_mu':>7} {'new_mu':>7} "
          f"{'old_lam':>8} {'new_lam':>8} {'mean_obs':>9}")
    print("  " + "-" * 70)

    for base in BASES:
        mus = []
        lams = []
        for des_rl in DESIGNED_RLS:
            obs = np.array(obs_by_base_level[base][des_rl])
            if len(obs) == 0:
                continue
            errors = obs - des_rl
            emp_mu = float(np.median(errors))
            emp_lam = float(np.mean(np.abs(errors - emp_mu)))
            mus.append(emp_mu)
            lams.append(emp_lam)

            old_mu, old_lam = OLD_LAPLACE[base]
            print(f"  {base:>4} {des_rl:>6} {len(obs):>7} {old_mu:>7.2f} {emp_mu:>7.2f} "
                  f"{old_lam:>8.2f} {emp_lam:>8.2f} {obs.mean():>9.2f}")

        # aggregate: one mu and lambda per base (median of per-level fits)
        agg_mu = float(np.median(mus))
        agg_lam = float(np.median(lams))
        new_laplace[base] = (agg_mu, agg_lam)

    print(f"\n  aggregated new params:")
    for base in BASES:
        old_mu, old_lam = OLD_LAPLACE[base]
        new_mu, new_lam = new_laplace[base]
        print(f"    {base}: mu {old_mu:+.2f} -> {new_mu:+.2f}, "
              f"lambda {old_lam:.2f} -> {new_lam:.2f}")

    # step 3: position-dependent effects
    print("\nSTEP 3: position-dependent effects")

    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt

    WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
            '#E69F00', '#56B4E9', '#F0E442']
    GR = 1.618

    pos_mean_error = []
    pos_std_error = []
    for ri in range(24):
        if not obs_by_position[ri]:
            pos_mean_error.append(0)
            pos_std_error.append(0)
            continue
        errors = [obs - des for des, obs in obs_by_position[ri]]
        pos_mean_error.append(np.mean(errors))
        pos_std_error.append(np.std(errors))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 7 / GR))
    ax1.bar(range(24), pos_mean_error, color=WONG[0], alpha=0.8)
    ax1.set_xlabel('Color position (0-23)')
    ax1.set_ylabel('Mean error [nt]')
    ax1.set_title('Mean HP length error vs position')
    ax1.axhline(0, color='black', lw=0.5)

    ax2.bar(range(24), pos_std_error, color=WONG[2], alpha=0.8)
    ax2.set_xlabel('Color position (0-23)')
    ax2.set_ylabel('Std error [nt]')
    ax2.set_title('HP length error variability vs position')

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'position_effects.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved position_effects.pdf")

    # step 4: designed-length-dependent distributions
    print("\nSTEP 4: designed-length-dependent distributions")

    from scipy.stats import laplace as laplace_dist

    fig, axes = plt.subplots(3, 4, figsize=(12, 12 / GR))
    for bi, base in enumerate(BASES):
        for li, des_rl in enumerate(DESIGNED_RLS):
            ax = axes[bi, li]
            obs = np.array(obs_by_base_level[base][des_rl])
            if len(obs) == 0:
                ax.text(0.5, 0.5, 'no data', transform=ax.transAxes, ha='center')
                continue

            # histogram of observed
            bins = np.arange(max(0, obs.min() - 1), obs.max() + 2) - 0.5
            ax.hist(obs, bins=bins, density=True, alpha=0.6, color=WONG[0],
                    label='observed')

            # old laplace
            x = np.linspace(max(0, obs.min() - 2), obs.max() + 2, 200)
            old_mu, old_lam = OLD_LAPLACE[base]
            old_pdf = laplace_dist.pdf(x, loc=des_rl + old_mu, scale=old_lam)
            ax.plot(x, old_pdf, '--', color=WONG[1], lw=1.5, label='old Laplace')

            # new laplace (per-level fit)
            errors = obs - des_rl
            emp_mu = float(np.median(errors))
            emp_lam = float(np.mean(np.abs(errors - emp_mu)))
            new_pdf = laplace_dist.pdf(x, loc=des_rl + emp_mu, scale=max(emp_lam, 0.1))
            ax.plot(x, new_pdf, '-', color=WONG[2], lw=1.5, label='new Laplace')

            ax.axvline(des_rl, color='black', ls=':', lw=0.8, alpha=0.5)
            ax.set_title(f'{base}, designed={des_rl}', fontsize=8)
            if bi == 2:
                ax.set_xlabel('Observed RL')
            if li == 0:
                ax.set_ylabel('Density')
            if bi == 0 and li == 0:
                ax.legend(fontsize=6)

    fig.suptitle('Observed run length distributions vs Laplace fits', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'level_distributions.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved level_distributions.pdf")

    # confusion between designed=3 and designed=4
    print("\n  3-vs-4 confusion check:")
    for base in BASES:
        obs3 = np.array(obs_by_base_level[base][3])
        obs4 = np.array(obs_by_base_level[base][4])
        if len(obs3) == 0 or len(obs4) == 0:
            continue
        # how often would designed=3 be decoded as intensity=1 (rl=4)?
        mu = new_laplace[base][0]
        wrong3 = np.sum(np.abs(obs3 - mu - 4) < np.abs(obs3 - mu - 3))
        wrong4 = np.sum(np.abs(obs4 - mu - 3) < np.abs(obs4 - mu - 4))
        print(f"    {base}: des=3 decoded as 4: {wrong3}/{len(obs3)} "
              f"({100*wrong3/len(obs3):.1f}%), "
              f"des=4 decoded as 3: {wrong4}/{len(obs4)} "
              f"({100*wrong4/len(obs4):.1f}%)")

    # step 5: recalibrate and re-run
    print("\nSTEP 5: recalibrate decoder with new params and re-run")

    # re-process all reads with new laplace params
    strip_color_obs_new = defaultdict(lambda: [[] for _ in range(24)])
    hmm_cache2 = {}
    n_reads = 0

    for seq in read_fastq(fastq):
        payload = orient_and_trim(seq)
        if payload is None or len(payload) < 100 or len(payload) > 400:
            continue

        si, hdist = classify_address_exact(payload, codebook)
        if hdist > 1 or si < 0:
            continue

        color_rls, color_bases = extract_color_rls(payload, addr_patterns[si], hmm_cache2)
        if color_rls is None:
            continue
        n_reads += 1

        for ri in range(24):
            if color_rls[ri] > 0:
                strip_color_obs_new[si][ri].append(
                    (color_rls[ri], color_bases[ri]))

    # reconstruct with new params
    recovered_new = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
    for si in range(N_STRIPS):
        fi = si // 128
        local = si % 128
        row = local // 4
        strip_col = local % 4
        for p in range(8):
            for ch in range(3):
                ri = p * 3 + ch
                obs_list = strip_color_obs_new[si][ri]
                if not obs_list:
                    continue
                votes = [nearest_intensity(rl, base, new_laplace)
                         for rl, base in obs_list]
                winner = Counter(votes).most_common(1)[0][0]
                col = strip_col * 8 + p
                recovered_new[fi, row, col, ch] = winner

    # accuracy
    total_correct_new = np.sum(gt_all == recovered_new)
    total_pixels = N_FRAMES * 32 * 32 * 3
    acc_new = total_correct_new / total_pixels

    # also compute with old params for comparison
    recovered_old = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
    for si in range(N_STRIPS):
        fi = si // 128
        local = si % 128
        row = local // 4
        strip_col = local % 4
        for p in range(8):
            for ch in range(3):
                ri = p * 3 + ch
                obs_list = strip_color_obs_new[si][ri]
                if not obs_list:
                    continue
                votes = [nearest_intensity(rl, base, OLD_LAPLACE)
                         for rl, base in obs_list]
                winner = Counter(votes).most_common(1)[0][0]
                col = strip_col * 8 + p
                recovered_old[fi, row, col, ch] = winner

    total_correct_old = np.sum(gt_all == recovered_old)
    acc_old = total_correct_old / total_pixels

    print(f"  reads used: {n_reads:,}")
    print(f"  old params accuracy: {acc_old:.4f}")
    print(f"  new params accuracy: {acc_new:.4f}")
    print(f"  improvement: {acc_new - acc_old:+.4f}")

    # per-frame
    print("\n  per-frame (new params):")
    for fi in range(N_FRAMES):
        c = np.sum(gt_all[fi] == recovered_new[fi])
        print(f"    frame {fi}: {c}/3072 = {c/3072:.4f}")

    perfect = sum(1 for fi in range(N_FRAMES) for r in range(32) for c in range(32)
                  if np.all(gt_all[fi, r, c] == recovered_new[fi, r, c]))
    print(f"  perfect pixels: {perfect}/{N_FRAMES*1024}")

    # save
    np.save(os.path.join(DATA_DIR, 'beaver_raw_recovered_calibrated.npy'), recovered_new)

    # step 6: what if mapping were {3,5,7,10}?
    print("\nSTEP 6: what-if analysis with mapping {3,5,7,10}")

    ALT_RLS = [3, 5, 7, 10]
    # remap ground truth: original mapping {3,4,7,10}, alt {3,5,7,10}
    # intensities 0,1,2,3 map to RL 3,5,7,10 instead of 3,4,7,10
    # we can't change the designed oligos, but we can check if the OBSERVED
    # run lengths are better separated with {3,5,7,10} boundaries

    # re-classify using alt boundaries with new laplace
    recovered_alt = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
    for si in range(N_STRIPS):
        fi = si // 128
        local = si % 128
        row = local // 4
        strip_col = local % 4
        for p in range(8):
            for ch in range(3):
                ri = p * 3 + ch
                obs_list = strip_color_obs_new[si][ri]
                if not obs_list:
                    continue
                votes = [nearest_intensity_custom(rl, base, new_laplace, ALT_RLS)
                         for rl, base in obs_list]
                winner = Counter(votes).most_common(1)[0][0]
                col = strip_col * 8 + p
                recovered_alt[fi, row, col, ch] = winner

    # the ground truth intensities (0-3) map to designed RLs {3,4,7,10}
    # but if we decode with {3,5,7,10} boundaries, intensity 1 (designed RL=4)
    # would need to be closer to 5 to be decoded as 1. so reads of designed=4
    # would mostly decode as intensity 0 (closer to 3 than to 5).
    # this means we can't directly compare to gt_all since the encoding used {3,4,7,10}.
    # instead, report what the single-read confusion would look like if we HAD used {3,5,7,10}.

    # simulate: for each designed RL, decode with alt boundaries
    print("  single-read accuracy if designed with {3,5,7,10}:")
    for base in BASES:
        correct = 0
        total = 0
        for li, des_rl in enumerate(DESIGNED_RLS):
            obs = obs_by_base_level[base][des_rl]
            for o in obs:
                # if the codec had used {3,5,7,10}, the designed RLs would be different
                # but the channel noise is the same. so we ask:
                # for obs drawn from designed=3, would it decode to level 0 with {3,5,7,10}?
                # for obs drawn from designed=4, would it decode to level 1 with {3,5,7,10}?
                # etc. but designed=4 doesn't exist in the alt mapping.
                # what we really want: if I=1 used RL=5 instead of RL=4,
                # would the reads (which would have RL~5+noise) decode better?
                pass
            total += len(obs)

    # better approach: for each designed level, shift the obs by the difference
    # designed=4 in old -> designed=5 in new means obs would be ~1 longer on average
    print("  hypothetical accuracy if I=1 had been encoded at RL=5 instead of 4:")
    for base in BASES:
        correct = 0; total = 0
        for li, (old_rl, new_rl) in enumerate(zip(DESIGNED_RLS, ALT_RLS)):
            obs = np.array(obs_by_base_level[base][old_rl])
            if len(obs) == 0:
                continue
            # shift observations: if designed was new_rl instead of old_rl,
            # observed would be obs + (new_rl - old_rl)
            shifted_obs = obs + (new_rl - old_rl)
            for o in shifted_obs:
                mu = new_laplace[base][0]
                corrected = o - mu
                pred = min(range(4), key=lambda i: abs(corrected - ALT_RLS[i]))
                if pred == li:
                    correct += 1
                total += 1
        print(f"    {base}: {correct}/{total} = {100*correct/max(total,1):.1f}%")

    # overall hypothetical
    all_correct = 0; all_total = 0
    for base in BASES:
        for li, (old_rl, new_rl) in enumerate(zip(DESIGNED_RLS, ALT_RLS)):
            obs = np.array(obs_by_base_level[base][old_rl])
            if len(obs) == 0:
                continue
            shifted = obs + (new_rl - old_rl)
            for o in shifted:
                mu = new_laplace[base][0]
                corrected = o - mu
                pred = min(range(4), key=lambda i: abs(corrected - ALT_RLS[i]))
                if pred == li:
                    all_correct += 1
                all_total += 1

    print(f"    overall: {all_correct}/{all_total} = {100*all_correct/max(all_total,1):.1f}%")

    # compare to current
    curr_correct = 0; curr_total = 0
    for base in BASES:
        for li, des_rl in enumerate(DESIGNED_RLS):
            obs = np.array(obs_by_base_level[base][des_rl])
            if len(obs) == 0:
                continue
            for o in obs:
                mu = new_laplace[base][0]
                corrected = o - mu
                pred = min(range(4), key=lambda i: abs(corrected - DESIGNED_RLS[i]))
                if pred == li:
                    curr_correct += 1
                curr_total += 1
    print(f"\n  current {'{3,4,7,10}'} single-read: {curr_correct}/{curr_total} "
          f"= {100*curr_correct/max(curr_total,1):.1f}%")
    print(f"  hypothetical {'{3,5,7,10}'} single-read: {all_correct}/{all_total} "
          f"= {100*all_correct/max(all_total,1):.1f}%")

    # save new params
    cal = {
        'old_laplace': {b: {'mu': m, 'lambda': l} for b, (m, l) in OLD_LAPLACE.items()},
        'new_laplace': {b: {'mu': m, 'lambda': l} for b, (m, l) in new_laplace.items()},
        'accuracy_old_params': float(acc_old),
        'accuracy_new_params': float(acc_new),
    }
    write_record(os.path.join(FIGURE_DIR, 'calibration.tsv'), cal)
    print(f"\n  saved calibration.tsv")


if __name__ == '__main__':
    main()
