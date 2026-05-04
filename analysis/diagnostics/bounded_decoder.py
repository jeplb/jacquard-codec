#!/usr/bin/env python3
"""
bounded color decoder: anchor from both ends to eliminate HMM drift.

address: run parser on first ~120 nt (unchanged, reliable for early bits)
color: find G-tail from 3' end, segment bounded substring into HP runs.
no HMM needed for color - just parse the bounded region directly.
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
DESIGNED_RLS = [3, 4, 7, 10]
N_STRIPS = 640
N_FRAMES = 5
RUN_PARSE_TRIM = 120
LAPLACE_PARAMS = {'A': (0, 0.75), 'C': (0, 0.85), 'T': (0, 0.77)}

COMP = str.maketrans('ACGT', 'TGCA')
def revcomp(seq): return seq.translate(COMP)[::-1]
FWD_CORE = FWD_PRIMER[5:20]
REV_CORE = REV_PRIMER[5:20]


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


def build_extended_pattern(codebook_bits):
    addr_pat = bits_to_pattern(codebook_bits)
    last = BASES.index(addr_pat[-1])
    cb = []
    for _ in range(24):
        last = (last + 1) % 3
        cb.append(BASES[last])
    return addr_pat + ''.join(cb)


def build_hmms(codebook):
    params = ProfileHMMParameters()
    params.hp_expected_length = float(ADDRESS_RL)
    hmms = {}
    for idx in range(len(codebook)):
        pattern = build_extended_pattern(codebook[idx])
        hmms[idx] = ProfileHMM(pattern, params)
    return hmms


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
    if seq[:60].find(revcomp(REV_CORE)) >= 0:
        payload = _trim_fwd_rev(revcomp(seq))
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


def classify_address(payload, codebook, addr_hmms):
    runs = parse_hp_runs(payload, RUN_PARSE_TRIM)
    runs = merge_noise_runs(runs)
    if len(runs) < 16:
        return -1, 'fail', -1
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
    if int(dists[best_idx]) <= 1:
        # estimate address end position in the payload
        addr_end = sum(l for _, l in runs[:16])
        # count skipped G's too
        act_seen = 0
        addr_end_pos = 0
        for i, b in enumerate(payload):
            if b in 'ACT':
                act_seen += 1
            # we've consumed enough ACT bases for 16 runs
            if act_seen >= addr_end:
                addr_end_pos = i + 1
                break
        return best_idx, 'stage1', addr_end_pos

    top5 = np.argsort(dists)[:5]
    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in payload.upper()], dtype=np.int32)
    if len(seq_idx) < 10:
        return -1, 'fail', -1
    best_score = -float('inf')
    best = int(top5[0])
    for ci in top5:
        _, score = addr_hmms[int(ci)].viterbi(seq_idx)
        if score > best_score:
            best_score = score; best = int(ci)
    # estimate address end for stage2 from runs
    addr_end = sum(l for _, l in runs[:16])
    act_seen = 0; addr_end_pos = 0
    for i, b in enumerate(payload):
        if b in 'ACT':
            act_seen += 1
        if act_seen >= addr_end:
            addr_end_pos = i + 1; break
    return best, 'stage2', addr_end_pos


# bounded color decoder

def find_gtail(payload):
    """find the start of the G-tail (3+ consecutive G's) from the 3' end."""
    # scan backward from end
    n = len(payload)
    i = n - 1
    while i >= 0 and payload[i] == 'G':
        i -= 1
    gtail_start = i + 1
    gtail_len = n - gtail_start
    if gtail_len >= 3:
        return gtail_start
    # no clean G-tail found, try searching for GGG+ anywhere in last 30 nt
    region = payload[-30:]
    for j in range(len(region) - 2):
        if region[j] == 'G' and region[j+1] == 'G' and region[j+2] == 'G':
            return n - 30 + j
    return -1


def segment_color_region(color_seq):
    """parse color region into HP runs, skipping G noise and merging."""
    runs = []
    for b in color_seq:
        if b not in 'ACT':
            continue
        if runs and runs[-1][0] == b:
            runs[-1] = (b, runs[-1][1] + 1)
        else:
            runs.append((b, 1))
    # merge singleton noise (same as address merge)
    return merge_noise_runs(runs)


def extract_color_bounded(payload, addr_end_pos, strip_cbases):
    """extract 24 color run lengths from bounded substring.

    returns list of 24 (run_length, base) tuples, or None if failed.
    """
    gtail_start = find_gtail(payload)
    if gtail_start < 0 or gtail_start <= addr_end_pos:
        return None

    color_seq = payload[addr_end_pos:gtail_start]
    runs = segment_color_region(color_seq)

    if len(runs) < 24:
        # not enough runs, pad with zeros
        result = []
        for ri in range(24):
            if ri < len(runs):
                result.append((runs[ri][1], runs[ri][0]))
            else:
                result.append((0, strip_cbases[ri]))
        return result

    # take first 24 runs
    return [(runs[ri][1], runs[ri][0]) for ri in range(24)]


# HMM color decoder (for comparison)

def extract_color_hmm(payload, addr_pattern, hmm_cache):
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
    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in payload[:280].upper()],
                       dtype=np.int32)
    if len(seq_idx) < 10:
        return None
    path, _ = hmm.viterbi(seq_idx)
    counts = {}
    for t, (st, pos, _) in enumerate(path):
        if st == StateType.MATCH and pos is not None:
            counts[pos] = counts.get(pos, 0) + 1
    return [counts.get(pos, 0) for pos in range(16, 40)]


def nearest_intensity(obs_rl, base):
    mu = LAPLACE_PARAMS.get(base, (0, 1.0))[0]
    corrected = obs_rl - mu
    dists = [abs(corrected - d) for d in DESIGNED_RLS]
    return dists.index(min(dists))


def read_fastq(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header: break
            seq = f.readline().strip()
            f.readline(); f.readline()
            yield seq


def main():
    if len(sys.argv) < 2:
        print("usage: bounded_decoder.py <fastq>")
        sys.exit(1)
    fastq = sys.argv[1]

    codebook = load_codebook(os.path.join(DATA_DIR, 'selected_codewords.tsv'))
    addr_patterns = {i: bits_to_pattern(codebook[i]) for i in range(len(codebook))}
    gt_all = np.stack([np.load(os.path.join(DATA_DIR, f'beaver_frame_{i}.npy'))
                       for i in range(N_FRAMES)])

    strip_cbases = {}
    for si in range(N_STRIPS):
        pat = addr_patterns[si]
        last = BASES.index(pat[-1])
        cb = []
        for _ in range(24):
            last = (last + 1) % 3
            cb.append(BASES[last])
        strip_cbases[si] = cb

    print("building HMMs...")
    addr_hmms = build_hmms(codebook)

    # process reads
    print("processing reads...")
    t0 = time.time()

    # observations for both decoders
    obs_bounded = defaultdict(lambda: [[] for _ in range(24)])
    obs_hmm = defaultdict(lambda: [[] for _ in range(24)])
    hmm_cache = {}

    stats = {'total': 0, 'classified': 0, 'gtail_found': 0,
             'bounded_ok': 0, 'hmm_ok': 0}

    for seq in read_fastq(fastq):
        payload = orient_and_trim(seq)
        if payload is None or len(payload) < 100 or len(payload) > 400:
            continue
        stats['total'] += 1

        si, method, addr_end_pos = classify_address(payload, codebook, addr_hmms)
        if si < 0 or si >= N_STRIPS:
            continue
        stats['classified'] += 1

        color_bases = strip_cbases[si]

        # bounded decoder
        if addr_end_pos > 0:
            result = extract_color_bounded(payload, addr_end_pos, color_bases)
            if result is not None:
                stats['bounded_ok'] += 1
                for ri in range(24):
                    rl, base = result[ri]
                    if rl > 0:
                        obs_bounded[si][ri].append((rl, base))

        # HMM decoder
        hmm_rls = extract_color_hmm(payload, addr_patterns[si], hmm_cache)
        if hmm_rls is not None:
            stats['hmm_ok'] += 1
            for ri in range(24):
                if hmm_rls[ri] > 0:
                    obs_hmm[si][ri].append((hmm_rls[ri], color_bases[ri]))

    elapsed = time.time() - t0
    print(f"  reads: {stats['total']:,}, classified: {stats['classified']:,}")
    print(f"  bounded ok: {stats['bounded_ok']:,}, HMM ok: {stats['hmm_ok']:,}")
    print(f"  time: {elapsed:.1f}s")

    # reconstruct both
    total_pixels = N_FRAMES * 32 * 32 * 3

    def reconstruct_and_analyze(strip_obs, label):
        recovered = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
        pos_correct = np.zeros(24, dtype=np.int64)
        pos_total = np.zeros(24, dtype=np.int64)
        conf_ch = {ch: np.zeros((4, 4), dtype=np.int64) for ch in range(3)}

        for si in range(N_STRIPS):
            fi = si // 128; local = si % 128
            row = local // 4; strip_col = local % 4
            for p in range(8):
                for ch in range(3):
                    ri = p * 3 + ch
                    obs = strip_obs[si][ri]
                    if not obs:
                        continue
                    votes = [nearest_intensity(rl, base) for rl, base in obs]
                    winner = Counter(votes).most_common(1)[0][0]
                    col = strip_col * 8 + p
                    recovered[fi, row, col, ch] = winner

                    true_int = int(gt_all[fi, row, col, ch])
                    pos_total[ri] += 1
                    if winner == true_int:
                        pos_correct[ri] += 1
                    conf_ch[ch][true_int, winner] += 1

        total_correct = np.sum(gt_all == recovered)
        acc = total_correct / total_pixels
        pos_acc = pos_correct / np.maximum(pos_total, 1)

        print(f"\n  {label}:")
        print(f"    overall: {total_correct}/{total_pixels} = {acc:.4f}")
        for fi in range(N_FRAMES):
            fa = np.sum(gt_all[fi] == recovered[fi]) / (32*32*3)
            print(f"    frame {fi}: {fa:.4f}")

        perfect = sum(1 for fi in range(N_FRAMES) for r in range(32) for c in range(32)
                      if np.all(gt_all[fi,r,c] == recovered[fi,r,c]))
        print(f"    perfect pixels: {perfect}/{N_FRAMES*1024}")

        return recovered, acc, pos_acc, conf_ch

    rec_bounded, acc_b, pos_acc_b, conf_b = reconstruct_and_analyze(obs_bounded, 'BOUNDED')
    rec_hmm, acc_h, pos_acc_h, conf_h = reconstruct_and_analyze(obs_hmm, 'HMM')

    # per-position comparison
    print("\n  per-position accuracy comparison:")
    print(f"  {'pos':>3}  {'bounded':>8}  {'HMM':>8}  {'delta':>8}")
    print("  " + "-" * 35)
    for ri in range(24):
        delta = pos_acc_b[ri] - pos_acc_h[ri]
        marker = " <<<" if delta > 0.05 else ""
        print(f"  {ri:>3}  {100*pos_acc_b[ri]:>7.1f}%  {100*pos_acc_h[ri]:>7.1f}%  "
              f"{100*delta:>+7.1f}%{marker}")

    print(f"\n  mean accuracy (positions 0-11):")
    print(f"    bounded: {100*pos_acc_b[:12].mean():.1f}%  HMM: {100*pos_acc_h[:12].mean():.1f}%")
    print(f"  mean accuracy (positions 12-23):")
    print(f"    bounded: {100*pos_acc_b[12:].mean():.1f}%  HMM: {100*pos_acc_h[12:].mean():.1f}%")

    # figures
    print("\ngenerating figures...")

    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt

    WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']
    GR = 1.618

    def cmy_to_rgb(img):
        return np.clip(1.0 - img.astype(float) / 3.0, 0, 1)

    # 1. position accuracy comparison
    fig, ax = plt.subplots(figsize=(7, 7 / GR))
    x = np.arange(24)
    ax.plot(x, 100 * pos_acc_h, 'o-', color=WONG[1], markersize=4, label='HMM (single-pass)')
    ax.plot(x, 100 * pos_acc_b, 's-', color=WONG[0], markersize=4, label='Bounded (anchor both ends)')
    ax.set_xlabel('Color position (0-23)')
    ax.set_ylabel('Consensus accuracy [%]')
    ax.set_title('Position-dependent accuracy')
    ax.legend(fontsize=8)
    ax.set_ylim(50, 105)
    for i in range(1, 8):
        ax.axvline(i * 3 - 0.5, color='gray', ls=':', lw=0.5, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'bounded_vs_hmm_by_position.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved bounded_vs_hmm_by_position.pdf")

    # 2. GT vs recovery comparison (both decoders)
    fig, axes = plt.subplots(3, 5, figsize=(10, 10 * 3 / (5 * GR)))
    for fi in range(5):
        axes[0, fi].imshow(cmy_to_rgb(gt_all[fi]), interpolation='nearest')
        axes[0, fi].set_title(f'Frame {fi}', fontsize=9)
        axes[0, fi].axis('off')
        axes[1, fi].imshow(cmy_to_rgb(rec_hmm[fi]), interpolation='nearest')
        axes[1, fi].axis('off')
        axes[2, fi].imshow(cmy_to_rgb(rec_bounded[fi]), interpolation='nearest')
        axes[2, fi].axis('off')
    axes[0, 0].set_ylabel('Ground truth', fontsize=9)
    axes[1, 0].set_ylabel(f'HMM ({100*acc_h:.1f}%)', fontsize=9)
    axes[2, 0].set_ylabel(f'Bounded ({100*acc_b:.1f}%)', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'bounded_vs_hmm_recovery.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved bounded_vs_hmm_recovery.pdf")

    # 3. confusion matrices (bounded)
    ch_names = ['C', 'M', 'Y']
    fig, axes = plt.subplots(1, 3, figsize=(11, 11 / (GR * 2)))
    for ci, ax in enumerate(axes):
        cm = conf_b[ci].astype(float)
        rs = cm.sum(axis=1, keepdims=True)
        rs[rs == 0] = 1
        cm_pct = cm / rs * 100
        im = ax.imshow(cm_pct, vmin=0, vmax=100, cmap='cividis')
        for i in range(4):
            for j in range(4):
                pct = cm_pct[i, j]
                color = 'white' if pct < 50 else 'black'
                ax.text(j, i, f'{pct:.0f}%', ha='center', va='center',
                        fontsize=8, color=color)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([f'I={i}' for i in range(4)], fontsize=7)
        ax.set_yticklabels([f'I={i}' for i in range(4)], fontsize=7)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ch_acc = np.trace(conf_b[ci]) / max(conf_b[ci].sum(), 1)
        ax.set_title(f'{ch_names[ci]} ({100*ch_acc:.1f}%)', fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Row-normalized [%]', fontsize=8)
    fig.suptitle('Post-consensus confusion (bounded decoder)', fontsize=10)
    fig.savefig(os.path.join(FIGURE_DIR, 'bounded_confusion.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved bounded_confusion.pdf")

    # save
    np.save(os.path.join(DATA_DIR, 'beaver_bounded_recovered.npy'), rec_bounded)

    summary = {
        'hmm_accuracy': float(acc_h),
        'bounded_accuracy': float(acc_b),
        'hmm_pos_accuracy': [float(a) for a in pos_acc_h],
        'bounded_pos_accuracy': [float(a) for a in pos_acc_b],
    }
    write_record(os.path.join(FIGURE_DIR, 'bounded_decoder.tsv'), summary)
    print("  saved bounded_decoder.tsv")


if __name__ == '__main__':
    main()
