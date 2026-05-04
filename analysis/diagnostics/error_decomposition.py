#!/usr/bin/env python3
"""
decompose the 13% pixel error into address vs color contributions.
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
VITERBI_TRIM = 280
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
    rc_rev_core = revcomp(REV_CORE)
    if seq[:60].find(rc_rev_core) >= 0:
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


def classify_address_full(payload, codebook, addr_hmms):
    """returns (strip_idx, method, hamming_dist, bits)."""
    runs = parse_hp_runs(payload, RUN_PARSE_TRIM)
    runs = merge_noise_runs(runs)
    if len(runs) < 16:
        return -1, 'fail', 99, None
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
    min_dist = int(dists[best_idx])

    if min_dist <= 1:
        return best_idx, 'stage1', min_dist, bits_arr

    top5 = np.argsort(dists)[:5]
    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in payload.upper()], dtype=np.int32)
    if len(seq_idx) < 10:
        return -1, 'fail', 99, bits_arr
    best_score = -float('inf')
    best = int(top5[0])
    for ci in top5:
        _, score = addr_hmms[int(ci)].viterbi(seq_idx)
        if score > best_score:
            best_score = score; best = int(ci)
    return best, 'stage2', min_dist, bits_arr


def extract_color_batch(seq_idxs, addr_pattern, hmm_cache):
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
    results = []
    for seq_idx in seq_idxs:
        if len(seq_idx) < 10:
            results.append(None); continue
        path, _ = hmm.viterbi(seq_idx)
        counts = {}
        for t, (st, pos, _) in enumerate(path):
            if st == StateType.MATCH and pos is not None:
                counts[pos] = counts.get(pos, 0) + 1
        results.append([counts.get(pos, 0) for pos in range(16, 40)])
    return results, color_bases


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
        print("usage: error_decomposition.py <fastq>")
        sys.exit(1)
    fastq = sys.argv[1]

    codebook = load_codebook(os.path.join(DATA_DIR, 'selected_codewords.tsv'))
    addr_patterns = {i: bits_to_pattern(codebook[i]) for i in range(len(codebook))}
    gt_all = np.stack([np.load(os.path.join(DATA_DIR, f'beaver_frame_{i}.npy'))
                       for i in range(N_FRAMES)])

    # precompute color bases per strip
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

    # per-strip data
    strip_seqidx = defaultdict(list)
    strip_hdists = defaultdict(list)  # hamming distances
    strip_bits = defaultdict(list)    # decoded bit arrays

    # per-read tracking for address analysis
    exact_reads = defaultdict(list)   # strip -> list of seq_idx (dist=0 only)
    nonexact_reads = defaultdict(list)  # strip -> list of (seq_idx, hdist)

    # bit error tracking
    bit_errors_at_dist1 = np.zeros(16, dtype=np.int64)
    bit_total_at_dist1 = 0

    for seq in read_fastq(fastq):
        payload = orient_and_trim(seq)
        if payload is None or len(payload) < 100 or len(payload) > 400:
            continue

        si, method, hdist, bits = classify_address_full(payload, codebook, addr_hmms)
        if si < 0 or si >= N_STRIPS:
            continue

        seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in payload[:VITERBI_TRIM].upper()],
                           dtype=np.int32)
        strip_seqidx[si].append(seq_idx)
        strip_hdists[si].append(hdist)

        if hdist == 0:
            exact_reads[si].append(seq_idx)
        else:
            nonexact_reads[si].append((seq_idx, hdist))

        # track bit errors for dist=1 reads
        if hdist == 1 and bits is not None:
            cw_bits = codebook[si]
            diff = bits != cw_bits
            bit_errors_at_dist1 += diff.astype(np.int64)
            bit_total_at_dist1 += 1

    print(f"  time: {time.time()-t0:.1f}s")
    coverages = np.array([len(strip_seqidx[i]) for i in range(N_STRIPS)])
    n_exact = sum(len(v) for v in exact_reads.values())
    n_nonexact = sum(len(v) for v in nonexact_reads.values())
    print(f"  exact (dist=0): {n_exact:,}")
    print(f"  non-exact (dist>=1): {n_nonexact:,}")

    # step 1: establish ground truth from exact-match reads
    print("\nSTEP 1: ground truth consensus from exact-match reads")

    hmm_cache = {}
    # exact-match consensus per strip per color position
    exact_consensus = {}  # (strip, ri) -> voted intensity
    exact_color_obs = defaultdict(lambda: [[] for _ in range(24)])

    for si in range(N_STRIPS):
        if not exact_reads[si]:
            continue
        results, color_bases = extract_color_batch(exact_reads[si], addr_patterns[si], hmm_cache)
        for color_rls in results:
            if color_rls is None:
                continue
            for ri in range(24):
                if color_rls[ri] > 0:
                    exact_color_obs[si][ri].append(
                        (color_rls[ri], color_bases[ri]))

        for ri in range(24):
            obs = exact_color_obs[si][ri]
            if obs:
                votes = [nearest_intensity(rl, base) for rl, base in obs]
                exact_consensus[(si, ri)] = Counter(votes).most_common(1)[0][0]

    # check non-exact reads against exact consensus
    n_misclass_suspect = 0
    n_nonexact_checked = 0
    for si in range(N_STRIPS):
        if not nonexact_reads[si]:
            continue
        # need exact consensus for this strip
        has_consensus = all((si, ri) in exact_consensus for ri in range(24))
        if not has_consensus:
            continue

        ne_seqidxs = [s for s, _ in nonexact_reads[si]]
        if not ne_seqidxs:
            continue
        results, color_bases = extract_color_batch(ne_seqidxs, addr_patterns[si], hmm_cache)
        for color_rls in results:
            if color_rls is None:
                continue
            n_nonexact_checked += 1
            disagreements = 0
            for ri in range(24):
                if color_rls[ri] <= 0:
                    continue
                pred = nearest_intensity(color_rls[ri], color_bases[ri])
                if (si, ri) in exact_consensus and pred != exact_consensus[(si, ri)]:
                    disagreements += 1
            # if disagrees on >20 of 24 positions, likely misclassified
            if disagreements > 20:
                n_misclass_suspect += 1

    print(f"  non-exact reads checked: {n_nonexact_checked:,}")
    print(f"  suspected misclassified (>20/24 disagree): {n_misclass_suspect:,} "
          f"({100*n_misclass_suspect/max(n_nonexact_checked,1):.1f}%)")

    # full color extraction for consensus
    print("\nFull color extraction...")
    strip_color_obs = defaultdict(lambda: [[] for _ in range(24)])
    for si in range(N_STRIPS):
        if not strip_seqidx[si]:
            continue
        results, color_bases = extract_color_batch(strip_seqidx[si], addr_patterns[si], hmm_cache)
        for color_rls in results:
            if color_rls is None:
                continue
            for ri in range(24):
                if color_rls[ri] > 0:
                    strip_color_obs[si][ri].append(
                        (color_rls[ri], color_bases[ri]))

    # reconstruct
    recovered = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
    for si in range(N_STRIPS):
        fi = si // 128; local = si % 128
        row = local // 4; strip_col = local % 4
        for p in range(8):
            for ch in range(3):
                ri = p * 3 + ch
                obs = strip_color_obs[si][ri]
                if not obs:
                    continue
                votes = [nearest_intensity(rl, base) for rl, base in obs]
                winner = Counter(votes).most_common(1)[0][0]
                recovered[fi, row, strip_col * 8 + p, ch] = winner

    total_pixels = N_FRAMES * 32 * 32 * 3
    total_correct = np.sum(gt_all == recovered)
    total_wrong = total_pixels - total_correct
    print(f"  overall accuracy: {total_correct}/{total_pixels} = {total_correct/total_pixels:.4f}")
    print(f"  total errors: {total_wrong}")

    # step 2: decompose errors
    print("\nSTEP 2: error decomposition")

    # for each wrong pixel, categorize
    cat_A = 0  # address error
    cat_B7 = 0  # color error at designed=7
    cat_B10 = 0  # color error at designed=10
    cat_C3 = 0  # color error at designed=3
    cat_C4 = 0  # color error at designed=4

    # per-channel error tracking
    ch_errors = {0: 0, 1: 0, 2: 0}
    ch_total = {0: 0, 1: 0, 2: 0}

    # per-position accuracy
    pos_correct = np.zeros(24, dtype=np.int64)
    pos_total = np.zeros(24, dtype=np.int64)

    # per-designed-level tracking
    level_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    level_total = {0: 0, 1: 0, 2: 0, 3: 0}
    level_confusion = {i: Counter() for i in range(4)}
    # per designed level per channel
    level_ch_correct = {(l, c): 0 for l in range(4) for c in range(3)}
    level_ch_total = {(l, c): 0 for l in range(4) for c in range(3)}

    # confusion matrices per channel (post-consensus)
    conf_ch = {ch: np.zeros((4, 4), dtype=np.int64) for ch in range(3)}

    for si in range(N_STRIPS):
        fi = si // 128; local = si % 128
        row = local // 4; strip_col = local % 4

        # check if this strip has exact-match consensus
        has_exact = all((si, ri) in exact_consensus for ri in range(24))

        for p in range(8):
            for ch in range(3):
                ri = p * 3 + ch
                col = strip_col * 8 + p
                true_int = int(gt_all[fi, row, col, ch])
                pred_int = int(recovered[fi, row, col, ch])
                des_rl = INTENSITY_TO_RL[true_int]

                ch_total[ch] += 1
                pos_total[ri] += 1
                level_total[true_int] += 1
                level_ch_total[(true_int, ch)] += 1
                conf_ch[ch][true_int, pred_int] += 1

                if pred_int == true_int:
                    pos_correct[ri] += 1
                    level_correct[true_int] += 1
                    level_ch_correct[(true_int, ch)] += 1
                    continue

                # error: classify it
                ch_errors[ch] += 1
                level_confusion[true_int][pred_int] += 1

                if has_exact:
                    # check if exact-match consensus agrees with ground truth
                    ec = exact_consensus.get((si, ri), -1)
                    if ec != true_int:
                        # exact-match reads also disagree with GT
                        # this is a color error, not address
                        pass

                # determine error category
                # check if exact consensus for this strip matches GT
                # if exact consensus != GT at most positions, address error
                if has_exact:
                    exact_matches_gt = sum(
                        1 for r in range(24)
                        if exact_consensus.get((si, r), -1) == int(
                            gt_all[fi, row, strip_col * 8 + r // 3, r % 3])
                    )
                    # actually this is per-pixel, let me simplify

                # simpler approach: if the exact-match consensus for this
                # specific position also got it wrong, it's a color error.
                # if exact consensus is right but full consensus is wrong,
                # address error (non-exact reads dragged the vote).
                ec = exact_consensus.get((si, ri), -1)

                if ec == -1:
                    # no exact consensus, can't determine
                    if des_rl == 10: cat_B10 += 1
                    elif des_rl == 7: cat_B7 += 1
                    elif des_rl == 4: cat_C4 += 1
                    else: cat_C3 += 1
                elif ec == true_int and pred_int != true_int:
                    # exact reads got it right, but full consensus wrong
                    # -> address errors in non-exact reads corrupted the vote
                    cat_A += 1
                else:
                    # exact consensus also wrong -> color error
                    if des_rl == 10: cat_B10 += 1
                    elif des_rl == 7: cat_B7 += 1
                    elif des_rl == 4: cat_C4 += 1
                    else: cat_C3 += 1

    print(f"\n  error source                  | pixels | % of total error")
    print(f"  ------------------------------|--------|------------------")
    print(f"  address misclassification     | {cat_A:>6} | {100*cat_A/max(total_wrong,1):>6.1f}%")
    print(f"  color error, designed=10      | {cat_B10:>6} | {100*cat_B10/max(total_wrong,1):>6.1f}%")
    print(f"  color error, designed=7       | {cat_B7:>6} | {100*cat_B7/max(total_wrong,1):>6.1f}%")
    print(f"  color error, designed=4       | {cat_C4:>6} | {100*cat_C4/max(total_wrong,1):>6.1f}%")
    print(f"  color error, designed=3       | {cat_C3:>6} | {100*cat_C3/max(total_wrong,1):>6.1f}%")
    print(f"  total                         | {total_wrong:>6} |")

    # step 3: per-position analysis
    print("\nSTEP 3: per-position accuracy")
    pos_acc = pos_correct / np.maximum(pos_total, 1)
    for ri in range(24):
        print(f"  pos {ri:>2}: {pos_correct[ri]}/{pos_total[ri]} = {pos_acc[ri]:.4f}")

    # step 4: per-designed-level
    print("\nSTEP 4: per-designed-level accuracy")
    for li in range(4):
        acc = level_correct[li] / max(level_total[li], 1)
        top_conf = level_confusion[li].most_common(1)
        conf_str = f"-> I={top_conf[0][0]} ({top_conf[0][1]}x)" if top_conf else "none"
        print(f"  I={li} (RL={DESIGNED_RLS[li]}): {level_correct[li]}/{level_total[li]} "
              f"= {100*acc:.1f}%, most common error: {conf_str}")
        for ch_name, ci in [('C', 0), ('M', 1), ('Y', 2)]:
            ch_acc = level_ch_correct[(li, ci)] / max(level_ch_total[(li, ci)], 1)
            print(f"    {ch_name}: {100*ch_acc:.1f}%")

    # step 5: address bit error pattern
    print("\nSTEP 5: address bit error rate (dist=1 reads)")
    if bit_total_at_dist1 > 0:
        for b in range(16):
            rate = bit_errors_at_dist1[b] / bit_total_at_dist1
            bar = '#' * int(rate * 100)
            print(f"  bit {b:>2}: {rate:.4f} {bar}")
    else:
        print("  no dist=1 reads")

    # step 7: figures
    print("\nSTEP 7: generating figures")

    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt

    WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#F0E442']
    GR = 1.618

    # 1. error_decomposition_pie.pdf
    labels = ['Address\nmisclassification', 'Color (RL=10)', 'Color (RL=7)',
              'Color (RL=4)', 'Color (RL=3)']
    sizes = [cat_A, cat_B10, cat_B7, cat_C4, cat_C3]
    colors = [WONG[3], WONG[1], WONG[4], WONG[0], WONG[2]]
    # filter out zeros
    nonzero = [(l, s, c) for l, s, c in zip(labels, sizes, colors) if s > 0]
    if nonzero:
        fig, ax = plt.subplots(figsize=(5, 5))
        ls, ss, cs = zip(*nonzero)
        wedges, texts, autotexts = ax.pie(
            ss, labels=ls, colors=cs, autopct='%1.1f%%',
            pctdistance=0.75, startangle=90)
        for t in autotexts:
            t.set_fontsize(8)
        for t in texts:
            t.set_fontsize(8)
        ax.set_title(f'Error decomposition ({total_wrong} total errors)', fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'error_decomposition_pie.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  saved error_decomposition_pie.pdf")

    # 2. accuracy_by_designed_level.pdf
    fig, ax = plt.subplots(figsize=(6, 6 / GR))
    x = np.arange(4)
    width = 0.2
    ch_names = ['C', 'M', 'Y']
    ch_colors = [WONG[0], WONG[1], WONG[2]]
    for ci, (cn, cc) in enumerate(zip(ch_names, ch_colors)):
        accs = [level_ch_correct[(li, ci)] / max(level_ch_total[(li, ci)], 1)
                for li in range(4)]
        ax.bar(x + ci * width - width, [100 * a for a in accs], width,
               label=cn, color=cc)
    ax.set_xticks(x)
    ax.set_xticklabels([f'I={i}\n(RL={DESIGNED_RLS[i]})' for i in range(4)])
    ax.set_ylabel('Consensus accuracy [%]')
    ax.set_title('Accuracy by designed intensity level')
    ax.legend()
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'accuracy_by_designed_level.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved accuracy_by_designed_level.pdf")

    # 3. accuracy_by_color_position.pdf
    fig, ax = plt.subplots(figsize=(7, 7 / GR))
    ax.bar(range(24), 100 * pos_acc, color=WONG[0], alpha=0.85, edgecolor='white')
    ax.axhline(100 * total_correct / total_pixels, color=WONG[1], ls='--', lw=1,
               label=f'Overall = {100*total_correct/total_pixels:.1f}%')
    ax.set_xlabel('Color position (0-23)')
    ax.set_ylabel('Consensus accuracy [%]')
    ax.set_title('Accuracy vs position within strand')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    # mark pixel boundaries
    for i in range(1, 8):
        ax.axvline(i * 3 - 0.5, color='gray', ls=':', lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'accuracy_by_color_position.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved accuracy_by_color_position.pdf")

    # 4. address_bit_error_rate.pdf
    if bit_total_at_dist1 > 0:
        fig, ax = plt.subplots(figsize=(6, 6 / GR))
        rates = bit_errors_at_dist1 / bit_total_at_dist1
        ax.bar(range(16), 100 * rates, color=WONG[0], edgecolor='white')
        ax.set_xlabel('Bit position (0=first transition, 15=last)')
        ax.set_ylabel('Error rate [%]')
        ax.set_title(f'Per-bit error rate (n={bit_total_at_dist1} dist-1 reads)')
        ax.set_xticks(range(16))
        fig.tight_layout()
        fig.savefig(os.path.join(FIGURE_DIR, 'address_bit_error_rate.pdf'),
                    dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("  saved address_bit_error_rate.pdf")

    # 5. color_confusion_by_channel.pdf
    fig, axes = plt.subplots(1, 3, figsize=(11, 11 / (GR * 2)))
    for ci, ax in enumerate(axes):
        cm = conf_ch[ci].astype(float)
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
        ch_acc = np.trace(conf_ch[ci]) / max(conf_ch[ci].sum(), 1)
        ax.set_title(f'{ch_names[ci]} ({100*ch_acc:.1f}%)', fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Row-normalized [%]', fontsize=8)
    fig.suptitle('Post-consensus confusion by channel', fontsize=10)
    fig.savefig(os.path.join(FIGURE_DIR, 'color_confusion_by_channel.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  saved color_confusion_by_channel.pdf")

    # save summary
    summary = {
        'total_errors': int(total_wrong),
        'error_address': int(cat_A),
        'error_color_rl10': int(cat_B10),
        'error_color_rl7': int(cat_B7),
        'error_color_rl4': int(cat_C4),
        'error_color_rl3': int(cat_C3),
        'position_accuracy': [float(a) for a in pos_acc],
        'level_accuracy': {str(i): float(level_correct[i]/max(level_total[i],1))
                           for i in range(4)},
    }
    write_record(os.path.join(FIGURE_DIR, 'error_decomposition.tsv'), summary)
    print("\n  saved error_decomposition.tsv")


if __name__ == '__main__':
    main()
