#!/usr/bin/env python3
"""
joint transition-and-run-length CMY codec encoder.

encodes a 5-frame 32x32 animation as 640 oligos. each oligo is one strip
of 8 contiguous pixels. 32 cols / 8 pixels per strip = 4 strips per row;
32 rows x 4 strips x 5 frames = 640 strips total.

strand layout:
  5'-[fwd_primer]-[16 address bases @ RL 5]-[8 x (C, M, Y) HP runs]-GGGG-[rev_primer]-3'

the 16 address bases encode strip index from a 640-entry codebook with
min hamming distance 4. the 24 colour runs use levels {3, 4, 7, 10} nt
(four intensities per channel = 4^3 = 64 colours per pixel). base
identity in the colour region cycles deterministically off the last
address base and carries no information.
"""

import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

FWD_PRIMER = "TGGCTCTATGTCGGGAAACCCCTAT"
REV_PRIMER = "ATGTGGCTCTGCCCATCGATTTGTC"
INTENSITY_TO_RL = {0: 3, 1: 4, 2: 7, 3: 10}
ADDRESS_RL = 5
GTAIL = "GGGG"
BASES = ['A', 'C', 'T']
PIXELS_PER_STRIP = 8
STRIPS_PER_ROW = 4  # 32 / 8


def load_codebook(path):
    """load selected codewords from TSV."""
    codewords = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 3:
                continue
            bits = [int(b) for b in parts[1]]
            codewords.append(bits)
    return codewords


def bits_to_dna(bits, start_base='A'):
    """encode bits as DNA transitions with fixed run length."""
    current = BASES.index(start_base)
    dna = []
    for b in bits:
        current = (current + 1) % 3 if b == 0 else (current + 2) % 3
        dna.append(BASES[current] * ADDRESS_RL)
    return ''.join(dna), BASES[current]


def color_to_dna(pixels_cmy, last_base):
    """encode 8 pixels x 3 channels = 24 HP runs.

    base order rotates continuously from last_base, advancing 1 step each run.
    """
    current = BASES.index(last_base)
    dna = []
    for pixel in pixels_cmy:
        for intensity in pixel:
            current = (current + 1) % 3
            rl = INTENSITY_TO_RL[int(intensity)]
            dna.append(BASES[current] * rl)
    return ''.join(dna)


def encode_animation(frame_paths, codebook_path, output_path):
    """encode multi-frame animation as FASTA."""
    codebook = load_codebook(codebook_path)

    frames = []
    for path in frame_paths:
        img = np.load(path)
        assert img.shape == (32, 32, 3) and img.dtype == np.uint8
        frames.append(img)

    n_frames = len(frames)
    n_strips = n_frames * 32 * STRIPS_PER_ROW
    assert len(codebook) >= n_strips, f"codebook {len(codebook)} < {n_strips} strips"

    print(f"encoding {n_frames} frames x 32x32 = {n_frames * 1024:,} pixels")
    print(f"  {PIXELS_PER_STRIP} pixels/strip, {STRIPS_PER_ROW} strips/row")
    print(f"  {n_strips} strips = {n_strips} oligos")

    lengths = []
    strip_idx = 0

    with open(output_path, 'w') as f:
        for frame_i, img in enumerate(frames):
            for row in range(32):
                for strip_col in range(STRIPS_PER_ROW):
                    col_start = strip_col * PIXELS_PER_STRIP
                    col_end = col_start + PIXELS_PER_STRIP

                    # extract 8 pixels
                    pixels = [tuple(img[row, c]) for c in range(col_start, col_end)]

                    # address
                    bits = codebook[strip_idx]
                    addr_dna, last_base = bits_to_dna(bits)

                    # color (24 runs)
                    color_dna = color_to_dna(pixels, last_base)

                    # assemble
                    oligo = FWD_PRIMER + addr_dna + color_dna + GTAIL + REV_PRIMER

                    # header (max 30 chars for oligo pool ordering)
                    header = f">babbage_{strip_idx:04d}"
                    f.write(f"{header}\n{oligo}\n")
                    lengths.append(len(oligo))
                    strip_idx += 1

    lengths = np.array(lengths)

    # color region stats
    color_lens = []
    for img in frames:
        for row in range(32):
            for strip_col in range(STRIPS_PER_ROW):
                col_start = strip_col * PIXELS_PER_STRIP
                cl = 0
                for c in range(col_start, col_start + PIXELS_PER_STRIP):
                    cl += sum(INTENSITY_TO_RL[int(v)] for v in img[row, c])
                color_lens.append(cl)
    color_lens = np.array(color_lens)

    addr_nt = 16 * ADDRESS_RL
    print(f"\noutput: {output_path}")
    print(f"  oligos: {len(lengths):,}")
    print(f"  oligo length: {lengths.min()}-{lengths.max()} nt (mean {lengths.mean():.0f})")
    print(f"  address: 16 x {ADDRESS_RL} = {addr_nt} nt")
    print(f"  color: {color_lens.min()}-{color_lens.max()} nt "
          f"(mean {color_lens.mean():.0f}, {PIXELS_PER_STRIP} pixels x 3 channels = 24 runs)")
    print(f"  G-tail: {len(GTAIL)} nt")
    print(f"  primers: {len(FWD_PRIMER)+len(REV_PRIMER)} nt")
    print(f"  payload: {addr_nt+color_lens.min()+len(GTAIL)}-"
          f"{addr_nt+color_lens.max()+len(GTAIL)} nt")

    n_pixels = n_frames * 1024
    total_bits = n_strips * 16 + n_pixels * 6
    print(f"\n  information: {n_strips} x 16 addr + {n_pixels} x 6 color = "
          f"{total_bits:,} bits = {total_bits//8:,} bytes")

    # HMM states
    n_match = 16 + 24 + 1  # address + color + G-tail
    print(f"  HMM: {n_match} match states (16 address + 24 color + 1 G-tail)")


def main():
    frame_paths = [os.path.join(DATA_DIR, f'beaver_frame_{i}.npy') for i in range(5)]
    missing = [p for p in frame_paths if not os.path.exists(p)]
    if missing:
        print(f"error: missing frames: {missing}")
        sys.exit(1)

    codebook_path = os.path.join(DATA_DIR, 'selected_codewords.tsv')
    if not os.path.exists(codebook_path):
        print("error: selected_codewords.tsv not found")
        sys.exit(1)

    output_path = os.path.join(DATA_DIR, 'beaver_animation_oligos.fasta')
    encode_animation(frame_paths, codebook_path, output_path)


if __name__ == '__main__':
    main()
