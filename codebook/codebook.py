#!/usr/bin/env python3
"""
codebook construction for the joint transition-and-run-length CMY codec.

greedy maximum minimum-distance selection of 1,012 codewords from
the 16-bit binary address space (65,536 candidates).

each codeword is a 16-bit binary string that maps to a 16-position
DNA transition sequence via the babbage cyclic rule.

scrambled with sha256(b'alohomora')[:2 bytes] before selection
to avoid trivial structure.
"""

import hashlib
import numpy as np
import time


def get_scramble_sequence(n_bits=16):
    """generate XOR scramble sequence from sha256(b'alohomora')."""
    digest = hashlib.sha256(b'alohomora').digest()
    bits = []
    for byte in digest[:n_bits // 8 + 1]:
        for i in range(7, -1, -1):
            bits.append((byte >> i) & 1)
    return np.array(bits[:n_bits], dtype=np.uint8)


def int_to_bits(val, n_bits=16):
    """convert integer to binary array."""
    return np.array([(val >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


def bits_to_int(bits):
    """convert binary array to integer."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming_distance(a, b):
    """hamming distance between two binary arrays."""
    return int(np.sum(a != b))


def build_codebook(n_codewords=1012, n_bits=16):
    """greedy maximum minimum-distance codebook construction.

    1. generate all 2^n_bits candidates
    2. XOR-scramble each with sha256(b'alohomora')
    3. greedily select codewords maximizing min pairwise hamming distance

    returns:
        codebook: array of shape (n_codewords, n_bits), scrambled binary codewords
        addresses: array of shape (n_codewords,), original integer addresses
        min_distance: minimum pairwise hamming distance in the codebook
    """
    scramble = get_scramble_sequence(n_bits)
    n_total = 2 ** n_bits

    print(f"codebook construction: selecting {n_codewords} from {n_total:,} candidates")
    print(f"  scramble key: sha256(b'alohomora') = {scramble.tolist()}")

    # precompute all scrambled codewords
    print("  generating candidates...", flush=True)
    all_codewords = np.zeros((n_total, n_bits), dtype=np.uint8)
    for i in range(n_total):
        all_codewords[i] = int_to_bits(i, n_bits) ^ scramble

    # greedy selection
    print("  greedy selection...", flush=True)
    t0 = time.time()

    # start with codeword 0 (scrambled)
    selected_idx = [0]
    # min distance from each candidate to the selected set
    min_dist_to_selected = np.full(n_total, n_bits + 1, dtype=np.int32)

    # update distances from first selected
    for j in range(n_total):
        d = hamming_distance(all_codewords[0], all_codewords[j])
        min_dist_to_selected[j] = d

    for step in range(1, n_codewords):
        # mask already selected
        min_dist_to_selected[selected_idx[-1]] = -1

        # find candidate with maximum min-distance to selected set
        best_idx = np.argmax(min_dist_to_selected)
        selected_idx.append(best_idx)

        # update min distances
        new_cw = all_codewords[best_idx]
        for j in range(n_total):
            if min_dist_to_selected[j] <= 0:
                continue
            d = hamming_distance(new_cw, all_codewords[j])
            if d < min_dist_to_selected[j]:
                min_dist_to_selected[j] = d

        if (step + 1) % 200 == 0 or step == n_codewords - 1:
            current_min = min(
                hamming_distance(all_codewords[selected_idx[i]], all_codewords[selected_idx[j]])
                for i in range(len(selected_idx))
                for j in range(i + 1, len(selected_idx))
                if j <= i + 5  # spot check, not full O(n^2)
            ) if len(selected_idx) > 1 else n_bits
            elapsed = time.time() - t0
            print(f"    {step + 1}/{n_codewords} selected ({elapsed:.1f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"  selection done in {elapsed:.1f}s")

    # extract selected codewords
    selected_idx = np.array(selected_idx)
    codebook = all_codewords[selected_idx]
    addresses = selected_idx  # original integer addresses

    # compute actual minimum pairwise hamming distance
    print("  computing minimum pairwise distance...", flush=True)
    min_distance = n_bits
    for i in range(n_codewords):
        for j in range(i + 1, n_codewords):
            d = hamming_distance(codebook[i], codebook[j])
            if d < min_distance:
                min_distance = d
                if min_distance <= 1:
                    break
        if min_distance <= 1:
            break

    print(f"  minimum pairwise hamming distance: {min_distance}")

    # distance distribution (sample)
    print("  distance distribution (sample of 10,000 pairs):")
    rng = np.random.RandomState(42)
    sample_dists = []
    for _ in range(10000):
        i, j = rng.randint(0, n_codewords, size=2)
        if i != j:
            sample_dists.append(hamming_distance(codebook[i], codebook[j]))
    sample_dists = np.array(sample_dists)
    print(f"    mean: {np.mean(sample_dists):.1f}")
    print(f"    median: {np.median(sample_dists):.0f}")
    print(f"    min (sample): {np.min(sample_dists)}")
    print(f"    max (sample): {np.max(sample_dists)}")

    return codebook, addresses, min_distance


def bits_to_dna(bits, start_base='A'):
    """convert binary codeword to DNA transition sequence.

    babbage cyclic rule over {A, C, T}:
      advance 1 = bit 0
      advance 2 = bit 1

    the cyclic order is A -> C -> T -> A.
    """
    bases = ['A', 'C', 'T']
    current = bases.index(start_base)
    dna = []
    for b in bits:
        if b == 0:
            current = (current + 1) % 3
        else:
            current = (current + 2) % 3
        dna.append(bases[current])
    return ''.join(dna)


def main():
    codebook, addresses, min_dist = build_codebook(n_codewords=1012, n_bits=16)

    # save
    np.savez('joint_codec/codebook.npz',
             codebook=codebook,
             addresses=addresses,
             min_distance=min_dist)

    # convert a few to DNA for inspection
    print("\nsample codewords:")
    print(f"  {'idx':>5} {'address':>7} {'bits':>18} {'DNA':>18}")
    for i in [0, 1, 2, 505, 1010, 1011]:
        bits_str = ''.join(str(b) for b in codebook[i])
        dna = bits_to_dna(codebook[i])
        print(f"  {i:>5} {addresses[i]:>7} {bits_str:>18} {dna:>18}")

    # pixel mapping
    print(f"\npixel mapping: 23 x 44 = 1,012 pixels")
    print(f"  pixel (0, 0) -> address {addresses[0]} -> {bits_to_dna(codebook[0])}")
    print(f"  pixel (21, 22) -> address {addresses[505]} -> {bits_to_dna(codebook[505])}")
    print(f"  pixel (43, 22) -> address {addresses[1011]} -> {bits_to_dna(codebook[1011])}")

    print(f"\ncodebook saved to joint_codec/codebook.npz")
    print(f"  shape: {codebook.shape}")
    print(f"  min pairwise hamming distance: {min_dist}")


if __name__ == '__main__':
    main()
