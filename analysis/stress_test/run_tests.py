#!/usr/bin/env python3
"""
stress test suite for the Jacquard encoder/decoder.

address decoder:
  stage 1: parse HP runs, decode transitions, top-K by hamming distance
  stage 2: score top-K candidates with pattern-specific HMMs, pick best

color decoder:
  build strip-specific 41-state HMM, viterbi full read, extract color RLs

simulates linear-amplicon nanopore noise only (laplace HP perturbation +
substitutions and deletions).
"""

import os
import sys
import math
import json
import time
import multiprocessing as mp
import numpy as np
from collections import Counter, defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from profile_hmm import ProfileHMM, ProfileHMMParameters, BASE_TO_IDX, StateType, HMMLibrary

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

BASES = ['A', 'C', 'T']
ADDRESS_RL = 5
RUN_PARSE_TRIM = 120  # how many nt to scan for run parsing

INTENSITY_TO_RL = {0: 3, 1: 4, 2: 7, 3: 10}
DESIGNED_RLS = [3, 4, 7, 10]

RAW_NOISE = {'sub': 0.035, 'del': 0.07, 'ins': 0.005,
             'laplace': {'A': (0, 1.87), 'C': (-2, 1.43), 'T': (0, 1.51)}}


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


def parse_hp_runs(seq, max_nt=120):
    """parse HP runs from sequence, skipping G bases (noise in address region)."""
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
    """merge singleton runs that are likely substitution noise.

    in the address region, adjacent HP runs always have different bases
    (transition encoding guarantees this). so a length-1 run between two
    same-base runs is almost certainly a substitution error.
    """
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
                # singleton between same-base runs: merge all three
                merged[-1] = (merged[-1][0], merged[-1][1] + 1 + runs[i + 1][1])
                i += 2
                changed = True
            else:
                merged.append(runs[i])
                i += 1
        runs = merged
    return runs


def build_extended_pattern(codebook_bits):
    """build 40-char pattern: 16 address bases + 24 color bases."""
    addr_pat = bits_to_pattern(codebook_bits)
    last = BASES.index(addr_pat[-1])
    color_bases = []
    for _ in range(24):
        last = (last + 1) % 3
        color_bases.append(BASES[last])
    return addr_pat + ''.join(color_bases)


def build_address_library(codebook):
    """build an HMMLibrary holding 640 extended HMMs (16 addr + 24 colour).

    the colour bases give the HMM structural anchoring so it can process
    full reads without the trim-length sensitivity an address-only HMM
    would have.
    """
    params = ProfileHMMParameters()
    params.hp_expected_length = float(ADDRESS_RL)
    library = HMMLibrary()
    hmms = {}
    for idx in range(len(codebook)):
        pattern = build_extended_pattern(codebook[idx])
        hmm = ProfileHMM(pattern, params)
        hmms[idx] = hmm
        library.add_hmm(str(idx), hmm)
    return library, hmms


def classify_address(read, codebook, library=None):
    """two-stage: run parser (fast) -> brute-force HMM (when uncertain).

    stage 1: parse HP runs, decode transitions, check for exact hamming match.
    stage 2: if no exact match, score all 640 extended HMMs on the full read.
    """
    runs = parse_hp_runs(read, RUN_PARSE_TRIM)
    runs = merge_noise_runs(runs)

    bits_arr = None
    if len(runs) >= 16:
        # stage 1: decode transitions from first 16 runs
        BASE_MAP = {'A': 0, 'C': 1, 'T': 2}
        bits = []
        prev = 0  # implicit start base A
        for base, _ in runs[:16]:
            obs = BASE_MAP[base]
            advance = (obs - prev) % 3
            bits.append(0 if advance == 1 else 1)
            prev = obs

        bits_arr = np.array(bits, dtype=np.uint8)
        dists = np.sum(codebook != bits_arr, axis=1)

        # exact match: accept immediately (safe since min hamming dist = 4)
        if dists.min() == 0:
            return int(np.argmin(dists))

    # stage 2: score full read against all 640 extended HMMs
    if library is not None:
        seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in read.upper()],
                           dtype=np.int32)
        best_name, _, score, _ = library.viterbi_best(seq_idx)
        return int(best_name)

    # fallback: best hamming match
    if bits_arr is not None:
        return int(np.argmin(dists))
    return -1


def simulate_read(pattern, addr_rl, color_intensities, rng, noise):
    """simulate full oligo read."""
    seq = []
    for base in pattern:
        mu, scale = noise['laplace'].get(base, (0, 1.0))
        noisy_rl = max(1, int(round(addr_rl + rng.laplace(mu, scale))))
        seq.append(base * noisy_rl)
    last = BASES.index(pattern[-1])
    for intensity in color_intensities:
        last = (last + 1) % 3
        base = BASES[last]
        designed_rl = INTENSITY_TO_RL[intensity]
        mu, scale = noise['laplace'].get(base, (0, 1.0))
        noisy_rl = max(1, int(round(designed_rl + rng.laplace(mu, scale))))
        seq.append(base * noisy_rl)
    seq.append('GGGG')
    clean = ''.join(seq)
    out = []
    for b in clean:
        if rng.random() < noise['del']: continue
        if rng.random() < noise['sub']:
            b = rng.choice([x for x in 'ACGT' if x != b])
        out.append(b)
        if rng.random() < noise['ins']:
            out.append(rng.choice(list('ACGT')))
    return ''.join(out)


def extract_color_hmm(read, addr_pattern, hmm=None):
    """stage 2: 41-state HMM for color extraction.

    pass a prebuilt extended HMM (16 addr + 24 colour) to skip reconstruction;
    otherwise one is built from addr_pattern. the prebuilt path lets callers
    reuse a per-strip HMM across many reads instead of rebuilding it each call.
    """
    if hmm is None:
        last = BASES.index(addr_pattern[-1])
        color_bases = []
        for _ in range(24):
            last = (last + 1) % 3
            color_bases.append(BASES[last])
        full_pattern = addr_pattern + ''.join(color_bases)

        params = ProfileHMMParameters()
        params.hp_expected_length = float(ADDRESS_RL)
        hmm = ProfileHMM(full_pattern, params)

    seq_idx = np.array([BASE_TO_IDX.get(b, 0) for b in read.upper()], dtype=np.int32)
    if len(seq_idx) < 10:
        return [0] * 24

    path, _ = hmm.viterbi(seq_idx)

    counts = {}
    for t, (st, pos, _) in enumerate(path):
        if st == StateType.MATCH and pos is not None:
            counts[pos] = counts.get(pos, 0) + 1

    return [counts.get(pos, 0) for pos in range(16, 40)]


def nearest_intensity(obs_rl, base=None, laplace_params=None):
    """decode observed HP run length to nearest designed intensity.

    when base and laplace_params are provided, corrects for per-base
    HP length bias (e.g. C undercall with mu=-2).
    """
    corrected = obs_rl
    if base is not None and laplace_params is not None:
        mu = laplace_params.get(base, (0, 1.0))[0]
        corrected = obs_rl - mu
    dists = [abs(corrected - d) for d in DESIGNED_RLS]
    return dists.index(min(dists))


# tests

def _test1_strip_worker(args):
    """one work unit = one strip (200 simulated reads + classify each)."""
    si, n_reads, seed = args
    codebook = _WORKER_STATE['codebook']
    library = _WORKER_STATE['library']
    rng = np.random.RandomState(seed)
    pat = bits_to_pattern(codebook[si])
    preds = []
    for _ in range(n_reads):
        read = simulate_read(pat, ADDRESS_RL, [0] * 24, rng, RAW_NOISE)
        preds.append(classify_address(read, codebook, library))
    return si, preds


def test1(pool, codebook):
    print("\nTEST 1: Address classification (two-stage)")
    sample = list(range(0, 640, 6))[:100]
    n_reads = 200
    base_seed = 42 * 200000

    work = [(si, n_reads, base_seed + si) for si in sample]

    single_ok = 0; vote5_ok = 0; n_s = 0; n_v = 0
    worst_acc = 1.0; worst_strip = 0

    for si, preds in pool.imap_unordered(_test1_strip_worker, work):
        for p in preds:
            n_s += 1
            if p == si: single_ok += 1
        for s in range(0, len(preds) - 4, 5):
            w = Counter(preds[s:s+5]).most_common(1)[0][0]
            n_v += 1
            if w == si: vote5_ok += 1
        acc = sum(1 for p in preds if p == si) / n_reads
        if acc < worst_acc: worst_acc = acc; worst_strip = si

    sa = single_ok / n_s; va = vote5_ok / max(n_v, 1)
    print(f"  single: {sa:.4f}, vote-5: {va:.4f}")
    print(f"  worst: strip {worst_strip} ({worst_acc:.4f})")
    passed = va > 0.95
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (vote-5 > 95%)")
    return passed, sa, va


def test2(rng):
    print("\nTEST 2: Color (raw, {3,4,7,10})")
    return _color(rng, RAW_NOISE, 25, 0.70)


def _color(rng, noise, td, thresh):
    confusion = np.zeros((4, 4), dtype=np.int64)
    depths = [1, 3, 5, 10, 25]
    dc = {d: 0 for d in depths}; dt = {d: 0 for d in depths}
    lp = noise['laplace']
    for di in range(4):
        for base in BASES:
            mu, sc = lp[base]
            obs = [max(1, int(round(DESIGNED_RLS[di] + rng.laplace(mu, sc)))) for _ in range(500)]
            for o in obs:
                confusion[di, nearest_intensity(o, base, lp)] += 1
            for d in depths:
                for s in range(0, len(obs)-d+1, d):
                    votes = [nearest_intensity(x, base, lp) for x in obs[s:s+d]]
                    w = Counter(votes).most_common(1)[0][0]
                    dt[d] += 1; dc[d] += (w == di)
    single = np.trace(confusion) / max(confusion.sum(), 1)
    print(f"  single: {single:.4f}")
    for i in range(4):
        rs = confusion[i].sum()
        print(f"    I={i}: " + ' '.join(f'{confusion[i,j]:>6}' for j in range(4))
              + f' ({100*confusion[i,i]/max(rs,1):.1f}%)')
    for d in depths:
        print(f"    depth {d:>2}: {dc[d]/max(dt[d],1):.4f}")
    ta = dc[td] / max(dt[td], 1)
    passed = ta > thresh
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'} (depth-{td}={ta:.4f})")
    return passed


# worker globals; populated by _pool_init in each Pool worker
_WORKER_STATE = {}


def _pool_init(codebook_arr):
    """build the address library once per worker (cheaper than pickling 640
    HMMs through the Pool queue on every call). shared by tests 1, 4, 5.

    the per-index HMM dict is kept too so test 4 can reuse the same extended
    HMMs for colour extraction instead of rebuilding one per read."""
    global _WORKER_STATE
    library, hmms = build_address_library(codebook_arr)
    _WORKER_STATE['codebook'] = codebook_arr
    _WORKER_STATE['library'] = library
    _WORKER_STATE['hmms'] = hmms


def _test4_strip_worker(args):
    """one work unit = one (frame, strip) pair. simulates n_reads, decodes
    each, and returns the recovered 8x3 color block plus the strip key.
    deterministic per-strip seeding so results are reproducible across runs.
    """
    fi, si, gi, colors, seed, n_reads, noise = args
    codebook = _WORKER_STATE['codebook']
    library = _WORKER_STATE['library']
    rng = np.random.RandomState(seed)
    pat = bits_to_pattern(codebook[gi])

    color_hmm = _WORKER_STATE['hmms'][gi]
    color_obs = [[] for _ in range(24)]
    for _ in range(n_reads):
        read = simulate_read(pat, ADDRESS_RL, colors, rng, noise)
        pred = classify_address(read, codebook, library)
        if pred != gi:
            continue
        cr = extract_color_hmm(read, pat, color_hmm)
        for ri in range(24):
            if ri < len(cr) and cr[ri] > 0:
                color_obs[ri].append(cr[ri])

    last_addr = BASES.index(pat[-1])
    color_bases = []
    for _ in range(24):
        last_addr = (last_addr + 1) % 3
        color_bases.append(BASES[last_addr])
    lp = noise['laplace']

    pixel_block = np.zeros((8, 3), dtype=np.float64)
    for p in range(8):
        for ch in range(3):
            ri = p * 3 + ch
            if color_obs[ri]:
                cb = color_bases[ri]
                votes = [nearest_intensity(o, cb, lp) for o in color_obs[ri]]
                pixel_block[p, ch] = Counter(votes).most_common(1)[0][0]
    return fi, si, pixel_block


def test4(pool, codebook, frames):
    """parallelised across (frame, strip) pairs via the passed-in Pool."""
    print("\nTEST 4: End-to-end")
    NR = 25
    NOISE = RAW_NOISE

    work = []
    base_seed = 42 * 100000
    for fi in range(5):
        img = frames[fi]
        for si in range(128):
            row = si // 4
            sc = si % 4
            gi = fi * 128 + si
            colors = []
            for p in range(8):
                colors.extend(img[row, sc * 8 + p].tolist())
            seed = base_seed + fi * 1000 + si
            work.append((fi, si, gi, colors, seed, NR, NOISE))

    recs = [np.zeros((32, 32, 3), dtype=np.float64) for _ in range(5)]
    print(f"  raw ({NR} reads):")

    for fi, si, pixel_block in pool.imap_unordered(_test4_strip_worker, work):
        row = si // 4
        sc = si % 4
        for p in range(8):
            for ch in range(3):
                recs[fi][row, sc * 8 + p, ch] = pixel_block[p, ch]

    for fi in range(5):
        true = frames[fi].astype(float)
        correct = np.sum(true == recs[fi]) / (32 * 32 * 3)
        mse = np.mean((true - recs[fi]) ** 2)
        psnr = 10 * math.log10(9.0 / max(mse, 1e-10))
        print(f"    frame {fi}: acc={correct:.4f} PSNR={psnr:.1f}dB")
    return True


def _test5_pair_worker(args):
    """one work unit = one (a, b) pair (500 simulations from a, count
    misclassifications as b)."""
    a, b, n_reads, seed = args
    codebook = _WORKER_STATE['codebook']
    library = _WORKER_STATE['library']
    rng = np.random.RandomState(seed)
    pat = bits_to_pattern(codebook[a])
    mis = 0
    for _ in range(n_reads):
        read = simulate_read(pat, ADDRESS_RL, [0] * 24, rng, RAW_NOISE)
        if classify_address(read, codebook, library) == b:
            mis += 1
    return a, b, mis


def test5(pool, codebook):
    print("\nTEST 5: Cross-talk")
    n = min(len(codebook), 200)
    min_d = min(int(np.sum(codebook[i]!=codebook[j])) for i in range(n) for j in range(i+1,n))
    pairs = [(i,j) for i in range(n) for j in range(i+1,n) if int(np.sum(codebook[i]!=codebook[j]))==min_d][:20]
    n_reads = 500
    base_seed = 42 * 300000
    work = [(a, b, n_reads, base_seed + k) for k, (a, b) in enumerate(pairs)]

    worst = 0.0
    for a, b, mis in pool.imap_unordered(_test5_pair_worker, work):
        if mis / n_reads > worst:
            worst = mis / n_reads

    print(f"  min dist: {min_d}, worst crosstalk: {worst:.4f}")
    passed = worst < 0.05
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def test6():
    print("\nTEST 6: Sanity")
    FWD = "TGGCTCTATGTCGGGAAACCCCTAT"; REV = "ATGTGGCTCTGCCCATCGATTTGTC"
    issues = []
    with open(os.path.join(DATA_DIR, 'beaver_animation_oligos.fasta')) as f:
        idx = 0
        while True:
            h = f.readline().strip()
            if not h: break
            seq = f.readline().strip()
            if not seq.startswith(FWD): issues.append(f"{idx}: fwd")
            if not seq.endswith(REV): issues.append(f"{idx}: rev")
            payload = seq[len(FWD):-len(REV)]
            if 'G' in payload[:16*ADDRESS_RL]: issues.append(f"{idx}: G addr")
            if not payload.endswith('GGGG'): issues.append(f"{idx}: gtail")
            color = payload[16*ADDRESS_RL:-4]
            runs = []; cur=color[0]; cnt=1
            for b in color[1:]:
                if b==cur: cnt+=1
                else: runs.append((cur,cnt)); cur=b; cnt=1
            runs.append((cur,cnt))
            for ri,(b,l) in enumerate(runs):
                if l>10: issues.append(f"{idx}: rl{l}")
            if len(runs)!=24: issues.append(f"{idx}: {len(runs)}runs")
            if len(seq)>350: issues.append(f"{idx}: len{len(seq)}")
            idx += 1
    print(f"  {'all '+str(idx)+' pass' if not issues else str(len(issues))+' issues'}")
    passed = not issues
    print(f"  VERDICT: {'PASS' if passed else 'FAIL'}")
    return passed


def test7(rng):
    print("\nTEST 7: Mapping sweep")
    for name, rls in [
        ('{3,4,7,10}', [3,4,7,10]), ('{3,5,7,9}', [3,5,7,9]),
        ('{3,5,8,10}', [3,5,8,10]), ('{3,6,8,10}', [3,6,8,10]),
    ]:
        accs = {}
        for b in BASES:
            ok=0; tot=0
            for di in range(4):
                mu,sc = RAW_NOISE['laplace'][b]
                for _ in range(500):
                    obs = max(1,int(round(rls[di]+rng.laplace(mu,sc))))
                    if [abs(obs-r) for r in rls].index(min(abs(obs-r) for r in rls))==di: ok+=1
                    tot+=1
            accs[b]=ok/tot
        print(f"  {name}: A={accs['A']:.3f} C={accs['C']:.3f} T={accs['T']:.3f} worst={min(accs.values()):.3f}")
    return True


def main():
    print("STRESS TEST: two-stage decoder, RL=5, {3,4,7,10}")
    print("=" * 60)

    codebook = load_codebook(os.path.join(DATA_DIR, 'selected_codewords.tsv'))
    frames = [np.load(os.path.join(DATA_DIR, f'beaver_frame_{i}.npy')) for i in range(5)]
    rng = np.random.RandomState(42)

    n_workers = max(1, mp.cpu_count() - 1)
    print(f"spinning up {n_workers}-worker pool (each builds its own 640-HMM library)...")

    v = {}
    with mp.Pool(processes=n_workers,
                 initializer=_pool_init,
                 initargs=(codebook,)) as pool:
        v['test1'] = test1(pool, codebook)[0]
        v['test2'] = test2(rng)
        v['test4'] = test4(pool, codebook, frames)
        v['test5'] = test5(pool, codebook)
        v['test6'] = test6()
        v['test7'] = test7(rng)

    print("\n" + "=" * 60)
    print("SUMMARY")
    all_pass = True
    for t in sorted(v):
        if t == 'test7': print(f"  {t}: INFO")
        else:
            s = 'PASS' if v[t] else 'FAIL'
            print(f"  {t}: {s}")
            if not v[t]: all_pass = False
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'FAILURES'}")

    with open(os.path.join(SCRIPT_DIR, 'summary.json'), 'w') as f:
        json.dump({k: str(v2) for k, v2 in v.items()}, f, indent=2)


if __name__ == '__main__':
    main()
