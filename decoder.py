#!/usr/bin/env python3
"""
decode real nanopore reads for the joint CMY codec beaver animation.

pipeline: trim primers -> classify address (run parser + viterbi fallback) ->
          extract color (bounded decoder: anchor address end + G-tail) ->
          consensus -> reconstruct frames
"""

import os, sys, gzip, time
import numpy as np
from collections import Counter, defaultdict

import numba
from numba import njit

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from tsv_records import write_record

from profile_hmm import ProfileHMM, ProfileHMMParameters, BASE_TO_IDX

DATA_DIR = os.path.join(REPO_ROOT, 'data')
FIGURE_DIR = REPO_ROOT

FWD_PRIMER = "TGGCTCTATGTCGGGAAACCCCTAT"
REV_PRIMER = "ATGTGGCTCTGCCCATCGATTTGTC"
BASES = ['A', 'C', 'T']
ADDRESS_RL = 5
INTENSITY_TO_RL = {0: 3, 1: 4, 2: 7, 3: 10}
DESIGNED_RLS = [3, 4, 7, 10]
N_STRIPS = 640
N_FRAMES = 5
RUN_PARSE_TRIM = 120
MIN_PAYLOAD = 100
MAX_PAYLOAD = 400
VITERBI_TRIM = 280  # truncate payload before viterbi (41 states, ~228 nt expected)
MAX_OBS_RL = 20  # max observed RL for channel histogram

# primer cores for fast str.find() matching (middle 15 of each 25-mer)
FWD_CORE = FWD_PRIMER[5:20]   # CTATGTCGGGAAACC
REV_CORE = REV_PRIMER[5:20]   # GCTCTGCCCATCGAT

COMP = str.maketrans('ACGT', 'TGCA')
def revcomp(seq):
    return seq.translate(COMP)[::-1]

RC_FWD = revcomp(FWD_PRIMER)
RC_REV = revcomp(REV_PRIMER)


# codebook and HMM library

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
    color_bases = []
    for _ in range(24):
        last = (last + 1) % 3
        color_bases.append(BASES[last])
    return addr_pat + ''.join(color_bases)


def build_hmms(codebook):
    """pre-build 640 extended HMMs for stage-2 viterbi fallback."""
    params = ProfileHMMParameters()
    params.hp_expected_length = float(ADDRESS_RL)
    hmms = {}
    for idx in range(len(codebook)):
        pattern = build_extended_pattern(codebook[idx])
        hmms[idx] = ProfileHMM(pattern, params)
    return hmms


# primer trimming

def _trim_fwd_rev(seq):
    """find FWD core in first 60 nt, REV core in last 60 nt. returns payload or None."""
    plen_f = len(FWD_PRIMER)
    plen_r = len(REV_PRIMER)
    head = seq[:60]
    pos_f = head.find(FWD_CORE)
    if pos_f < 0:
        return None
    # the core starts at offset 5 within the primer
    payload_start = pos_f - 5 + plen_f
    if payload_start < 0:
        payload_start = pos_f + len(FWD_CORE)

    tail = seq[-60:]
    pos_r = tail.find(REV_CORE)
    if pos_r >= 0:
        # rev core is at offset 5 within REV_PRIMER
        payload_end = len(seq) - 60 + pos_r - 5
        if payload_end > payload_start:
            return seq[payload_start:payload_end]
    # no rev primer found, take everything after fwd
    return seq[payload_start:]


def orient_and_trim(seq):
    """orient read and trim primers using fast core matching."""
    # try forward: FWD_CORE in head
    payload = _trim_fwd_rev(seq)
    if payload is not None:
        return payload, 'fwd'

    # try reverse: RC_REV starts the read -> revcomp then trim
    # check for rc(REV) core in first 60
    rc_rev_core = revcomp(REV_CORE)
    if seq[:60].find(rc_rev_core) >= 0:
        flipped = revcomp(seq)
        payload = _trim_fwd_rev(flipped)
        if payload is not None:
            return payload, 'rev'

    return None, None


# address classification

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


def _estimate_addr_end(payload, runs):
    """estimate the position in payload where the 16th address run ends."""
    target_act = sum(l for _, l in runs[:16])
    act_count = 0
    for i, b in enumerate(payload):
        if b in 'ACT':
            act_count += 1
        if act_count >= target_act:
            return i + 1
    return len(payload)


def classify_address(payload, codebook, addr_hmms):
    """two-stage address classifier.

    returns (strip_idx, method, addr_end_pos).
    addr_end_pos is the estimated position where the address region ends.
    """
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
    min_dist = int(dists[best_idx])

    addr_end = _estimate_addr_end(payload, runs)

    # stage 1: unambiguous match (within correction radius)
    if min_dist <= 1:
        return best_idx, 'stage1', addr_end

    # stage 2: ambiguous zone, score top-5 by viterbi
    top5 = np.argsort(dists)[:5]
    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in payload.upper()],
                       dtype=np.int32)
    best_score = -float('inf')
    best = int(top5[0])
    for ci in top5:
        _, score = addr_hmms[int(ci)].viterbi(seq_idx)
        if score > best_score:
            best_score = score
            best = int(ci)
    return best, 'stage2', addr_end


# bounded color extraction

def _find_gtail(payload):
    """find the start of the G-tail (3+ consecutive G's) from the 3' end."""
    n = len(payload)
    i = n - 1
    while i >= 0 and payload[i] == 'G':
        i -= 1
    gtail_start = i + 1
    if n - gtail_start >= 3:
        return gtail_start
    # fallback: search for GGG+ in last 30 nt
    region = payload[-30:]
    for j in range(len(region) - 2):
        if region[j] == 'G' and region[j+1] == 'G' and region[j+2] == 'G':
            return n - 30 + j
    return -1


def _segment_color_runs(color_seq):
    """parse color region into HP runs, skip G noise, merge singletons."""
    runs = []
    for b in color_seq:
        if b not in 'ACT':
            continue
        if runs and runs[-1][0] == b:
            runs[-1] = (b, runs[-1][1] + 1)
        else:
            runs.append((b, 1))
    return merge_noise_runs(runs)


def extract_color_bounded(payload, addr_end_pos, color_bases):
    """extract 24 color run lengths from the bounded substring between
    address end and G-tail. no HMM needed.

    returns list of 24 (run_length, base) or None.
    """
    gtail = _find_gtail(payload)
    if gtail < 0 or gtail <= addr_end_pos:
        return None

    color_seq = payload[addr_end_pos:gtail]
    runs = _segment_color_runs(color_seq)

    if len(runs) < 20:
        return None

    result = []
    for ri in range(24):
        if ri < len(runs):
            result.append((runs[ri][1], runs[ri][0]))
        else:
            result.append((0, color_bases[ri]))
    return result


def build_empirical_channel(strip_color_obs, designed_map, codebook):
    """build P(obs=k | designed=d, base=b) from exact-match reads.

    returns log_channel[base][designed_rl] = array of log-probs for k in [0..MAX_OBS_RL].
    """
    hist = {b: {d: np.zeros(MAX_OBS_RL + 1, dtype=np.float64)
                for d in DESIGNED_RLS} for b in BASES}

    for si in range(N_STRIPS):
        for ri in range(24):
            des_int = designed_map.get((si, ri), -1)
            if des_int < 0:
                continue
            des_rl = INTENSITY_TO_RL[des_int]
            for obs_rl, obs_base in strip_color_obs[si][ri]:
                k = min(obs_rl, MAX_OBS_RL)
                hist[obs_base][des_rl][k] += 1

    # add pseudocount, normalize, take log
    log_channel = {}
    for b in BASES:
        log_channel[b] = {}
        for d in DESIGNED_RLS:
            hist[b][d] += 0.5
            hist[b][d] /= hist[b][d].sum()
            log_channel[b][d] = np.log(hist[b][d] + 1e-30)

    return log_channel


def ml_classify_single(obs_rl, base, log_channel):
    """ML classification of a single read's HP observation."""
    k = min(obs_rl, MAX_OBS_RL)
    return int(np.argmax([log_channel[base][DESIGNED_RLS[di]][k] for di in range(4)]))


def soft_likelihood_aggregate(obs_list, log_channel):
    """sum log-likelihoods across reads, return argmax class.

    obs_list: list of (obs_rl, base) tuples.
    returns (predicted_class, ll_gap between top two classes).

    kept as the reference implementation; production callers use the
    packed/njit version below.
    """
    scores = np.zeros(4, dtype=np.float64)
    for obs_rl, base in obs_list:
        k = min(obs_rl, MAX_OBS_RL)
        for di in range(4):
            scores[di] += log_channel[base][DESIGNED_RLS[di]][k]
    best = int(np.argmax(scores))
    sorted_s = np.sort(scores)[::-1]
    gap = sorted_s[0] - sorted_s[1]
    return best, gap


def pack_log_channel(log_channel):
    """convert dict-of-dict-of-list log_channel into a contiguous float64 array.

    indexing: arr[base_idx, designed_idx, k] = log P(obs=k | designed, base)
    where base_idx follows BASE_TO_IDX (4-way ACGT) and designed_idx follows
    the order of DESIGNED_RLS. unused base slots (G in this codec) are filled
    with a very negative sentinel.
    """
    n_bases = len(BASE_TO_IDX)
    n_designed = len(DESIGNED_RLS)
    arr = np.full((n_bases, n_designed, MAX_OBS_RL + 1), -1e30, dtype=np.float64)
    for base, by_designed in log_channel.items():
        bi = BASE_TO_IDX[base]
        for di, d in enumerate(DESIGNED_RLS):
            arr[bi, di, :] = by_designed[d]
    return arr


def pack_obs_list(obs_list):
    """convert list of (obs_rl, base) tuples into parallel int8 arrays.

    obs_rls clipped to MAX_OBS_RL, bases int-encoded via BASE_TO_IDX.
    """
    n = len(obs_list)
    obs_rls = np.empty(n, dtype=np.int8)
    bases = np.empty(n, dtype=np.int8)
    for i, (rl, b) in enumerate(obs_list):
        obs_rls[i] = min(rl, MAX_OBS_RL)
        bases[i] = BASE_TO_IDX[b]
    return obs_rls, bases


@njit(cache=True)
def soft_likelihood_aggregate_njit(obs_rls, bases, log_channel_arr):
    """njit version of soft_likelihood_aggregate.

    obs_rls:         int8 array of clipped observed RLs
    bases:           int8 array of base indices (BASE_TO_IDX)
    log_channel_arr: float64 array of shape (n_bases, n_designed, max_obs_rl+1)

    returns (best_idx, gap) where gap = best_score - second_best_score.
    best/second-best computed in two passes to avoid sort allocation.
    """
    n_designed = log_channel_arr.shape[1]
    scores = np.zeros(n_designed, dtype=np.float64)
    n = obs_rls.shape[0]
    for i in range(n):
        b = bases[i]
        k = obs_rls[i]
        for di in range(n_designed):
            scores[di] += log_channel_arr[b, di, k]

    # pass 1: best
    best_idx = 0
    best_val = scores[0]
    for di in range(1, n_designed):
        if scores[di] > best_val:
            best_val = scores[di]
            best_idx = di

    # pass 2: second-best (excluding best_idx)
    second_val = -np.inf
    for di in range(n_designed):
        if di == best_idx:
            continue
        if scores[di] > second_val:
            second_val = scores[di]

    return best_idx, best_val - second_val


def bootstrap_accuracy_vs_depth(packed_strip_obs, log_channel_arr, gt_all,
                                recovered, total_pixels, depths_to_test,
                                n_reps=100, seed=42):
    """per-pixel accuracy vs subsampled read depth (soft-LL bootstrap).

    a cell with n_obs <= depth uses all its reads every replicate, so its
    prediction is constant and equals the all-reads consensus already stored
    in `recovered`. those cells are filled once per depth instead of being
    re-scored n_reps times. only subsampled cells (n_obs > depth) draw from
    `rng`, in the same (strip, ri) order as the naive loop, so the random
    stream -- and therefore every reported number -- is identical.

    returns {depth: (mean_acc, lo_ci, hi_ci)}.
    """
    rng = np.random.RandomState(seed)
    depth_results = {}

    for depth in depths_to_test:
        # base frame: full-coverage cells take the all-reads consensus,
        # other cells stay 0; subsampled cells are overwritten each replicate.
        base = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
        subsample_cells = []  # (frame_i, row, col, ch, obs_rls, bases_arr)
        for strip_idx, slot in packed_strip_obs.items():
            frame_i = strip_idx // 128
            local = strip_idx % 128
            row = local // 4
            strip_col = local % 4
            for ri, (obs_rls, bases_arr) in slot.items():
                p_idx = ri // 3
                ch = ri % 3
                col = strip_col * 8 + p_idx
                if obs_rls.shape[0] <= depth:
                    base[frame_i, row, col, ch] = recovered[frame_i, row, col, ch]
                else:
                    subsample_cells.append(
                        (frame_i, row, col, ch, obs_rls, bases_arr))

        accs = []
        for rep in range(n_reps):
            rec_sub = base.copy()
            for frame_i, row, col, ch, obs_rls, bases_arr in subsample_cells:
                idxs = rng.choice(obs_rls.shape[0], depth, replace=False)
                pred, _ = soft_likelihood_aggregate_njit(
                    obs_rls[idxs], bases_arr[idxs], log_channel_arr
                )
                rec_sub[frame_i, row, col, ch] = pred
            accs.append(np.sum(gt_all == rec_sub) / total_pixels)

        accs = np.array(accs)
        lo, hi = np.percentile(accs, [2.5, 97.5])
        depth_results[depth] = (accs.mean(), lo, hi)
        print(f"    depth {depth:>2}: {accs.mean():.4f} [{lo:.4f}, {hi:.4f}]")

    return depth_results


# FASTQ reading

def read_fastq(path):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header:
                break
            seq = f.readline().strip()
            f.readline()
            f.readline()
            yield header, seq


# designed oligo lengths (from encoder)

def get_designed_lengths(codebook):
    """compute designed oligo payload length for each strip."""
    frames = []
    for i in range(N_FRAMES):
        path = os.path.join(DATA_DIR, f'beaver_frame_{i}.npy')
        frames.append(np.load(path))

    lengths = []
    strip_idx = 0
    for frame_i, img in enumerate(frames):
        for row in range(32):
            for strip_col in range(4):
                col_start = strip_col * 8
                cl = 0
                for c in range(col_start, col_start + 8):
                    cl += sum(INTENSITY_TO_RL[int(v)] for v in img[row, c])
                # payload = address + color + gtail
                payload_len = 16 * ADDRESS_RL + cl + 4
                lengths.append(payload_len)
                strip_idx += 1
    return np.array(lengths)


# main pipeline

def main():
    if len(sys.argv) < 2:
        print("usage: python decoder.py <reads.fastq.gz>")
        sys.exit(1)
    fastq = sys.argv[1]

    codebook_path = os.path.join(DATA_DIR, 'selected_codewords.tsv')
    codebook = load_codebook(codebook_path)
    addr_patterns = {i: bits_to_pattern(codebook[i]) for i in range(len(codebook))}

    print(f"input: {fastq}")
    print(f"codebook: {len(codebook)} codewords\n")

    # build HMMs for stage-2 fallback
    t0 = time.time()
    print("building 640 extended HMMs (16 addr + 24 color)...")
    addr_hmms = build_hmms(codebook)
    print(f"  done in {time.time()-t0:.1f}s\n")

    # step 1: primer trimming
    print("STEP 1: primer trimming")
    t0 = time.time()

    stats = {'total': 0, 'primer_found': 0, 'length_pass': 0,
             'fwd': 0, 'rev': 0, 'too_short': 0, 'too_long': 0}
    payloads = []  # (payload, read_id)
    payload_lengths = []

    for header, seq in read_fastq(fastq):
        stats['total'] += 1
        payload, orient = orient_and_trim(seq)
        if payload is None:
            continue
        stats['primer_found'] += 1
        if orient == 'fwd':
            stats['fwd'] += 1
        else:
            stats['rev'] += 1

        plen = len(payload)
        if plen < MIN_PAYLOAD:
            stats['too_short'] += 1
            continue
        if plen > MAX_PAYLOAD:
            stats['too_long'] += 1
            continue
        stats['length_pass'] += 1
        payloads.append(payload)
        payload_lengths.append(plen)

    payload_lengths = np.array(payload_lengths)
    elapsed = time.time() - t0

    print(f"  total reads: {stats['total']:,}")
    print(f"  primer found: {stats['primer_found']:,} "
          f"(fwd={stats['fwd']:,} rev={stats['rev']:,})")
    print(f"  too short (<{MIN_PAYLOAD}): {stats['too_short']:,}")
    print(f"  too long (>{MAX_PAYLOAD}): {stats['too_long']:,}")
    print(f"  passing filter: {stats['length_pass']:,}")
    print(f"  payload length: min={payload_lengths.min()} "
          f"median={np.median(payload_lengths):.0f} "
          f"max={payload_lengths.max()}")
    print(f"  time: {elapsed:.1f}s\n")

    # step 2: address classification
    print("STEP 2: address classification")
    t0 = time.time()

    # store payloads per strip, separated by confidence
    # stage 1 (dist<=1): used for color voting
    # stage 2 (dist>=2): counted for coverage but excluded from color voting
    strip_payloads = defaultdict(list)      # stage 1 only (for color)
    strip_payloads_all = defaultdict(list)  # all classified (for coverage stats)
    cls_stats = {'stage1': 0, 'stage2': 0, 'fail': 0}

    for i, payload in enumerate(payloads):
        strip_idx, method, addr_end_pos = classify_address(payload, codebook, addr_hmms)
        cls_stats[method] += 1
        if 0 <= strip_idx < N_STRIPS and addr_end_pos > 0:
            strip_payloads_all[strip_idx].append((payload, addr_end_pos))
            if method == 'stage1':
                strip_payloads[strip_idx].append((payload, addr_end_pos))
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{len(payloads):,} classified ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    coverages = np.array([len(strip_payloads_all[i]) for i in range(N_STRIPS)])
    color_coverages = np.array([len(strip_payloads[i]) for i in range(N_STRIPS)])
    total_classified = int(coverages.sum())

    print(f"  classified: {total_classified:,}")
    print(f"  stage 1 (dist<=1): {cls_stats['stage1']:,} "
          f"({100*cls_stats['stage1']/max(len(payloads),1):.1f}%)")
    print(f"  stage 2 (top-5 viterbi): {cls_stats['stage2']:,} "
          f"({100*cls_stats['stage2']/max(len(payloads),1):.1f}%)")
    print(f"  failed: {cls_stats['fail']:,}")
    print(f"  strips with data: {np.sum(coverages > 0)}/{N_STRIPS}")
    print(f"  coverage (all): min={coverages.min()} median={np.median(coverages):.0f} "
          f"mean={coverages.mean():.1f} max={coverages.max()}")
    print(f"  coverage (stage 1 only): min={color_coverages.min()} "
          f"median={np.median(color_coverages):.0f} "
          f"mean={color_coverages.mean():.1f} max={color_coverages.max()}")
    print(f"  strips with 0 coverage: {np.sum(color_coverages == 0)}")
    print(f"  time: {elapsed:.1f}s\n")

    # step 3: color extraction (bounded decoder)
    print("STEP 3: color extraction (bounded decoder)")
    t0 = time.time()

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

    strip_color_obs = defaultdict(lambda: [[] for _ in range(24)])
    n_extracted = 0

    for strip_idx in range(N_STRIPS):
        color_bases = strip_cbases[strip_idx]
        for payload, addr_end_pos in strip_payloads[strip_idx]:
            result = extract_color_bounded(payload, addr_end_pos, color_bases)
            if result is None:
                continue
            for ri in range(24):
                rl, base = result[ri]
                if rl > 0:
                    strip_color_obs[strip_idx][ri].append((rl, base))
            n_extracted += 1
        if (strip_idx + 1) % 100 == 0:
            print(f"  {strip_idx+1}/640 strips ({time.time()-t0:.1f}s)")

    elapsed = time.time() - t0
    print(f"  reads with color extracted: {n_extracted:,}")
    print(f"  time: {elapsed:.1f}s\n")

    # load ground truth
    gt_frames = []
    for fi in range(N_FRAMES):
        gt_frames.append(np.load(os.path.join(DATA_DIR, f'beaver_frame_{fi}.npy')))
    gt_all = np.stack(gt_frames)  # (5, 32, 32, 3)

    # precompute designed intensities for channel building
    designed_map = {}
    for si in range(N_STRIPS):
        fi = si // 128; local = si % 128
        row = local // 4; strip_col = local % 4
        for ri in range(24):
            p = ri // 3; ch = ri % 3
            col = strip_col * 8 + p
            designed_map[(si, ri)] = int(gt_all[fi, row, col, ch])

    # build empirical channel from exact-match reads only
    # first identify which strips had exact matches
    exact_obs = defaultdict(lambda: [[] for _ in range(24)])
    for i, payload in enumerate(payloads):
        runs = merge_noise_runs(parse_hp_runs(payload, RUN_PARSE_TRIM))
        if len(runs) < 16:
            continue
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
        best = int(np.argmin(dists))
        if int(dists[best]) != 0:
            continue
        # this is an exact-match read - use for channel estimation
        if best in strip_payloads:
            for payload_s, ae in strip_payloads[best]:
                if payload_s == payload:
                    gt = _find_gtail(payload_s)
                    if gt > 0 and gt > ae:
                        runs_c = _segment_color_runs(payload_s[ae:gt])
                        if len(runs_c) >= 24:
                            for ri in range(24):
                                exact_obs[best][ri].append(
                                    (runs_c[ri][1], runs_c[ri][0]))
                    break

    print("  building empirical channel from exact-match reads...")
    log_channel = build_empirical_channel(exact_obs, designed_map, codebook)

    # pack once: log_channel into a 3D float64 array, and every populated
    # obs_list into parallel int8 arrays. the bootstrap calls each obs_list
    # ~700 times (7 depth tiers x 100 replicates), so the packing cost is
    # heavily amortised.
    log_channel_arr = pack_log_channel(log_channel)
    packed_strip_obs = {}
    for strip_idx in range(N_STRIPS):
        ros = strip_color_obs.get(strip_idx)
        if not ros:
            continue
        slot = {}
        for ri in range(24):
            obs_list = ros[ri]
            if obs_list:
                slot[ri] = pack_obs_list(obs_list)
        if slot:
            packed_strip_obs[strip_idx] = slot

    # single-read intensity confusion (ML classifier)
    confusion_per_ch = {ch: np.zeros((4, 4), dtype=np.int64) for ch in range(3)}

    for strip_idx in range(N_STRIPS):
        frame_i = strip_idx // 128
        local = strip_idx % 128
        row = local // 4
        strip_col = local % 4
        for ri in range(24):
            p = ri // 3; ch = ri % 3
            col = strip_col * 8 + p
            true_int = int(gt_all[frame_i, row, col, ch])
            for obs_rl, obs_base in strip_color_obs[strip_idx][ri]:
                pred_int = ml_classify_single(obs_rl, obs_base, log_channel)
                confusion_per_ch[ch][true_int, pred_int] += 1

    print("  single-read intensity confusion (ML classifier):")
    conf_total = sum(confusion_per_ch.values())
    total_obs = conf_total.sum()
    correct_obs = np.trace(conf_total)
    print(f"    total observations: {total_obs:,}")
    print(f"    correct: {correct_obs:,} ({100*correct_obs/max(total_obs,1):.1f}%)")
    for ch_name, ch_idx in [('C', 0), ('M', 1), ('Y', 2)]:
        cm = confusion_per_ch[ch_idx]
        acc = np.trace(cm) / max(cm.sum(), 1)
        print(f"    {ch_name}: {100*acc:.1f}%")

    # step 4: soft likelihood consensus
    print("\nSTEP 4: soft likelihood consensus")

    recovered = np.zeros((N_FRAMES, 32, 32, 3), dtype=np.uint8)
    total_pixels = N_FRAMES * 32 * 32 * 3

    for strip_idx, slot in packed_strip_obs.items():
        frame_i = strip_idx // 128
        local = strip_idx % 128
        row = local // 4
        strip_col = local % 4
        for ri, (obs_rls, bases_arr) in slot.items():
            p = ri // 3
            ch = ri % 3
            pred, _ = soft_likelihood_aggregate_njit(
                obs_rls, bases_arr, log_channel_arr
            )
            col = strip_col * 8 + p
            recovered[frame_i, row, col, ch] = pred

    print("  per-frame accuracy:")
    for fi in range(N_FRAMES):
        correct = np.sum(gt_frames[fi] == recovered[fi])
        total = 32 * 32 * 3
        print(f"    frame {fi}: {correct}/{total} = {correct/total:.4f}")

    total_correct = np.sum(gt_all == recovered)
    overall_acc = total_correct / total_pixels
    print(f"    overall: {total_correct}/{total_pixels} = {overall_acc:.4f}")

    perfect_pixels = 0
    for fi in range(N_FRAMES):
        for r in range(32):
            for c in range(32):
                if np.all(gt_all[fi, r, c] == recovered[fi, r, c]):
                    perfect_pixels += 1
    print(f"  perfectly recovered pixels: {perfect_pixels}/{N_FRAMES*1024}")

    np.save(os.path.join(DATA_DIR, 'beaver_raw_recovered.npy'), recovered)
    print(f"  saved: beaver_raw_recovered.npy")

    # accuracy vs depth (soft likelihood subsampling)
    print("\n  accuracy vs depth (100 bootstrap replicates, soft LL):")
    depths_to_test = [1, 3, 5, 10, 15, 20, 25]
    depth_results = bootstrap_accuracy_vs_depth(
        packed_strip_obs, log_channel_arr, gt_all, recovered,
        total_pixels, depths_to_test, n_reps=100, seed=42,
    )

    # step 5: figures
    print("\nSTEP 5: generating figures")

    import matplotlib
    matplotlib.use('Agg')
    matplotlib.rcParams['font.sans-serif'] = ['Arial']
    matplotlib.rcParams['pdf.fonttype'] = 42
    import matplotlib.pyplot as plt

    # wong palette
    WONG = ['#0072B2', '#D55E00', '#009E73', '#CC79A7',
            '#E69F00', '#56B4E9', '#F0E442']
    GR = 1.618

    def cmy_to_rgb(img_cmy):
        return np.clip(1.0 - img_cmy.astype(float) / 3.0, 0, 1)

    # figure 1: beaver_raw_recovery.pdf
    fig, axes = plt.subplots(2, 5, figsize=(10, 10 / GR))
    for fi in range(5):
        rgb = cmy_to_rgb(recovered[fi])
        axes[0, fi].imshow(rgb, interpolation='nearest')
        axes[0, fi].set_title(f'Frame {fi}', fontsize=9)
        axes[0, fi].axis('off')
        # 10x magnified (just zoom in on center 10x10)
        axes[1, fi].imshow(rgb, interpolation='nearest')
        axes[1, fi].set_xlim(6, 26)
        axes[1, fi].set_ylim(26, 6)
        axes[1, fi].axis('off')
    axes[0, 0].set_ylabel('Native', fontsize=9)
    axes[1, 0].set_ylabel('Zoomed', fontsize=9)
    fig.suptitle('Recovered animation (raw nanopore)', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_raw_recovery.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figure 2: beaver_ground_truth_vs_raw.pdf
    fig, axes = plt.subplots(2, 5, figsize=(10, 10 / GR))
    for fi in range(5):
        gt_rgb = cmy_to_rgb(gt_all[fi])
        rec_rgb = cmy_to_rgb(recovered[fi])
        axes[0, fi].imshow(gt_rgb, interpolation='nearest')
        axes[0, fi].set_title(f'Frame {fi}', fontsize=9)
        axes[0, fi].axis('off')
        axes[1, fi].imshow(rec_rgb, interpolation='nearest')
        axes[1, fi].axis('off')
    axes[0, 0].set_ylabel('Ground truth', fontsize=9)
    axes[1, 0].set_ylabel('Recovered', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_ground_truth_vs_raw.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figure 3: beaver_pixel_diff.pdf
    fig, axes = plt.subplots(1, 5, figsize=(10, 10 / (GR * 3)))
    for fi in range(5):
        diff = (gt_all[fi] != recovered[fi]).any(axis=2).astype(float)
        # red where different, white where same
        img = np.ones((32, 32, 3))
        img[diff > 0] = [1, 0.2, 0.2]
        axes[fi].imshow(img, interpolation='nearest')
        n_err = int(diff.sum())
        axes[fi].set_title(f'Frame {fi} ({n_err} errors)', fontsize=8)
        axes[fi].axis('off')
    fig.suptitle('Pixel errors (red = incorrect)', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_pixel_diff.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figure 4: beaver_coverage_distribution.pdf
    fig, ax = plt.subplots(figsize=(5, 5 / GR))
    ax.hist(coverages, bins=30, color=WONG[0], edgecolor='white', alpha=0.85)
    ax.axvline(coverages.mean(), color=WONG[1], ls='--', lw=1.5,
               label=f'Mean = {coverages.mean():.1f}')
    ax.axvline(np.median(coverages), color=WONG[2], ls='--', lw=1.5,
               label=f'Median = {np.median(coverages):.0f}')
    ax.set_xlabel('Reads per strip')
    ax.set_ylabel('Count')
    ax.set_title('Coverage distribution')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_coverage_distribution.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figure 5: beaver_coverage_vs_length.pdf
    designed_lengths = get_designed_lengths(codebook)
    fig, ax = plt.subplots(figsize=(5, 5 / GR))
    ax.scatter(designed_lengths, coverages, s=8, alpha=0.5, color=WONG[0])
    ax.set_xlabel('Designed oligo payload length [nt]')
    ax.set_ylabel('Read coverage')
    ax.set_title('Coverage vs oligo length')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_coverage_vs_length.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figure 6: beaver_confusion_matrices.pdf
    fig, axes = plt.subplots(1, 3, figsize=(11, 11 / (GR * 2)))
    ch_names = ['C', 'M', 'Y']
    for ci, ax in enumerate(axes):
        cm = confusion_per_ch[ci].astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_pct = cm / row_sums * 100
        im = ax.imshow(cm_pct, vmin=0, vmax=100, cmap='cividis')
        for i in range(4):
            for j in range(4):
                pct = cm_pct[i, j]
                color = 'white' if pct < 50 else 'black'
                ax.text(j, i, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=8, color=color)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([f'I={i}' for i in range(4)], fontsize=7)
        ax.set_yticklabels([f'I={i}' for i in range(4)], fontsize=7)
        ax.set_xlabel('Predicted', fontsize=8)
        ax.set_ylabel('True', fontsize=8)
        ch_acc = np.trace(confusion_per_ch[ci]) / max(confusion_per_ch[ci].sum(), 1)
        ax.set_title(f'{ch_names[ci]} ({100*ch_acc:.1f}%)', fontsize=9)
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Row-normalized [%]', fontsize=8)
    fig.suptitle('Single-read intensity confusion', fontsize=10)
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_confusion_matrices.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    # figure 7: beaver_accuracy_vs_depth.pdf
    fig, ax = plt.subplots(figsize=(5, 5 / GR))
    ds = sorted(depth_results.keys())
    means = [depth_results[d][0] for d in ds]
    los = [depth_results[d][1] for d in ds]
    his = [depth_results[d][2] for d in ds]
    ax.fill_between(ds, los, his, alpha=0.2, color=WONG[0])
    ax.plot(ds, means, 'o-', color=WONG[0], markersize=5)
    ax.axhline(overall_acc, color=WONG[1], ls='--', lw=1,
               label=f'Actual depth ({coverages.mean():.0f}x) = {overall_acc:.3f}')
    ax.set_xlabel('Read depth')
    ax.set_ylabel('Per-pixel accuracy')
    ax.set_title('Accuracy vs sequencing depth')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, 'beaver_accuracy_vs_depth.pdf'),
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"  saved 7 figures to {FIGURE_DIR}/")

    # step 6: summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  total reads after filtering:    {stats['length_pass']:,}")
    print(f"  mean coverage per strip:        {coverages.mean():.1f}")
    print(f"  address classification:")
    print(f"    stage 1 (dist<=1):            {cls_stats['stage1']:,} "
          f"({100*cls_stats['stage1']/max(len(payloads),1):.1f}%)")
    print(f"    stage 2 (top-5 viterbi):      {cls_stats['stage2']:,} "
          f"({100*cls_stats['stage2']/max(len(payloads),1):.1f}%)")

    # estimate address accuracy from majority agreement
    agree_count = 0
    total_reads_for_agree = 0
    for strip_idx in range(N_STRIPS):
        n_reads = len(strip_payloads[strip_idx])
        if n_reads < 3:
            continue
        agree_count += n_reads
        total_reads_for_agree += n_reads
    print(f"    reads in classified strips:   {agree_count:,}")

    print(f"  per-pixel accuracy:")
    for d in [1, 3, 5]:
        if d in depth_results:
            print(f"    vote-{d}:                      {depth_results[d][0]:.4f}")
    print(f"    at actual depth ({coverages.mean():.0f}x):       {overall_acc:.4f}")

    print(f"  per-channel accuracy (at actual depth):")
    for ch_name, ch_idx in [('C', 0), ('M', 1), ('Y', 2)]:
        # compute consensus accuracy per channel
        ch_correct = 0; ch_total = 0
        for fi in range(N_FRAMES):
            ch_correct += np.sum(gt_all[fi, :, :, ch_idx] == recovered[fi, :, :, ch_idx])
            ch_total += 32 * 32
        print(f"    {ch_name}: {ch_correct}/{ch_total} = {ch_correct/ch_total:.4f}")

    # best/worst frame
    frame_accs = []
    for fi in range(N_FRAMES):
        fa = np.sum(gt_all[fi] == recovered[fi]) / (32 * 32 * 3)
        frame_accs.append(fa)
    print(f"  best frame:                     {np.argmax(frame_accs)} "
          f"({max(frame_accs):.4f})")
    print(f"  worst frame:                    {np.argmin(frame_accs)} "
          f"({min(frame_accs):.4f})")
    print(f"  perfectly recovered pixels:     {perfect_pixels}/{N_FRAMES*1024}")

    # save summary table
    summary = {
        'total_reads': stats['total'],
        'reads_after_filter': stats['length_pass'],
        'mean_coverage': float(coverages.mean()),
        'median_coverage': float(np.median(coverages)),
        'overall_accuracy': float(overall_acc),
        'perfect_pixels': perfect_pixels,
        'frame_accuracies': [float(a) for a in frame_accs],
        'depth_results': {str(d): {'mean': float(m), 'lo': float(l), 'hi': float(h)}
                          for d, (m, l, h) in depth_results.items()},
    }
    write_record(os.path.join(FIGURE_DIR, 'decode_summary.tsv'), summary)
    print(f"\n  saved: decode_summary.tsv")


if __name__ == '__main__':
    main()
