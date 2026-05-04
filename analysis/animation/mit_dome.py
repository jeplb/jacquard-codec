#!/usr/bin/env python3
"""
32x32 pixel art of the MIT great dome from the Killian Court view.
reference: frontal view with dome on top, ionic columns, flanking trees, green lawn.
4-level CMY color.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['pdf.fonttype'] = 42

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, 'data')

W, H = 32, 32
img = np.zeros((H, W, 3), dtype=np.uint8)


def sp(r, c, cmy):
    if 0 <= r < H and 0 <= c < W:
        img[r, c] = cmy


def fill_rect(r0, r1, c0, c1, cmy):
    for r in range(r0, r1):
        for c in range(c0, c1):
            sp(r, c, cmy)


# color palette (CMY)
SKY_LIGHT = (1, 0, 0)     # light blue
SKY_MED = (2, 0, 0)       # medium blue
CLOUD = (0, 0, 0)         # white
STONE_LIGHT = (0, 1, 1)   # warm beige/cream (light)
STONE_MED = (1, 1, 1)     # medium gray stone
STONE_DARK = (1, 2, 1)    # darker warm stone
DOME_GREEN = (1, 1, 0)    # patina green-gray for dome cap
DOME_STONE = (0, 1, 1)    # warm stone
COLUMN_WHITE = (0, 0, 0)  # white columns
FAINT_CYAN = (1, 0, 0)   # faint cyan background (replaces pure white)
COLUMN_SHADOW = (1, 1, 1) # shadow between columns
WINDOW_DARK = (2, 2, 1)   # dark windows/glass
TREE_DARK = (2, 0, 1)     # dark green foliage
TREE_LIGHT = (1, 0, 1)    # lighter green foliage
TRUNK = (1, 2, 2)         # brown trunk
GRASS = (1, 0, 1)         # bright green lawn
GRASS_DARK = (2, 0, 2)    # darker green
MAROON = (1, 3, 3)        # MIT maroon for letters

# sky (rows 0-8)
# blue sky gradient: deeper at top, lighter near horizon
for r in range(9):
    for c in range(W):
        if r < 3:
            sp(r, c, SKY_MED)   # deeper blue at top (C=2)
        else:
            sp(r, c, SKY_LIGHT) # lighter blue lower (C=1)

# clouds (puffy shapes, not lines)
# cloud 1: rows 1-2, cols 8-16
for r, c in [(1, 10), (1, 11), (1, 12), (1, 13),
             (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
             (1, 9), (1, 14)]:
    sp(r, c, CLOUD)

# cloud 2: rows 2-3, cols 20-27
for r, c in [(2, 22), (2, 23), (2, 24),
             (3, 21), (3, 22), (3, 23), (3, 24), (3, 25),
             (2, 21), (2, 25), (3, 20), (3, 26)]:
    sp(r, c, CLOUD)

# cloud 3: rows 4-5, cols 4-10
for r, c in [(4, 6), (4, 7), (4, 8),
             (5, 5), (5, 6), (5, 7), (5, 8), (5, 9),
             (4, 5), (4, 9)]:
    sp(r, c, CLOUD)

# deep blue accent in upper corners (C=3)
for r in range(2):
    sp(r, 0, (3, 0, 0))
    sp(r, 31, (3, 0, 0))


# dome (rows 5-13)
dome_cx = 16
dome_r = 7
dome_base = 13
bldg_left = 5
bldg_right = 27

# dome fill: grayscale 3D shading (light from upper right)
for r in range(5, dome_base + 1):
    for c in range(dome_cx - dome_r, dome_cx + dome_r + 1):
        dx = c - dome_cx
        dy = dome_base - r
        if dx * dx + dy * dy <= dome_r * dome_r:
            nx = dx / dome_r
            ny = dy / dome_r
            shade = 0.5 + 0.35 * nx + 0.25 * ny
            shade = max(0.0, min(1.0, shade))
            level = int(round(shade * 3))
            level = max(1, min(3, level))
            sp(r, c, (level, level, level))

# dome cap: lighter gray at the top (rows 5-7)
for r in range(5, 8):
    for c in range(dome_cx - dome_r, dome_cx + dome_r + 1):
        dx = c - dome_cx
        dy = dome_base - r
        if dx * dx + dy * dy <= dome_r * dome_r:
            sp(r, c, (2, 2, 2))  # medium gray cap

# dome outline
for angle_deg in range(0, 181):
    angle = np.radians(angle_deg)
    x = dome_cx + dome_r * np.cos(angle)
    y = dome_base - dome_r * np.sin(angle)
    sp(int(round(y)), int(round(x)), (3, 3, 3))

# horizontal bands
for c in range(dome_cx - dome_r + 1, dome_cx + dome_r):
    dx = c - dome_cx
    if dx * dx + 16 <= dome_r * dome_r:
        sp(9, c, (2, 2, 2))
    if dx * dx + 4 <= dome_r * dome_r:
        sp(11, c, (2, 2, 2))

# MIT letters on dome face (rows 9-11, centered, maroon)
# 3 rows tall for legibility
mit_left = dome_cx - 4

# M: 5 wide x 3 tall
mx = mit_left
sp(9, mx, MAROON); sp(9, mx+4, MAROON)        # top: two outer verticals
sp(9, mx+1, MAROON); sp(9, mx+3, MAROON)      # top: inner diagonals
sp(10, mx, MAROON); sp(10, mx+2, MAROON); sp(10, mx+4, MAROON)  # mid: verticals + center V
sp(11, mx, MAROON); sp(11, mx+4, MAROON)       # bottom: two verticals

# I: 1 wide x 3 tall
ix = mx + 6
sp(9, ix, MAROON); sp(10, ix, MAROON); sp(11, ix, MAROON)

# T: 3 wide x 3 tall
tx = ix + 2
sp(9, tx, MAROON); sp(9, tx+1, MAROON); sp(9, tx+2, MAROON)  # top bar
sp(10, tx+1, MAROON)  # stem
sp(11, tx+1, MAROON)  # stem


# entablature / frieze (rows 13-14)
fill_rect(13, 15, bldg_left, bldg_right, (2, 2, 2))
# frieze detail line
for c in range(bldg_left, bldg_right):
    sp(14, c, (1, 1, 1))


# columns section (rows 15-24)

# dark windows behind columns
fill_rect(15, 25, bldg_left, bldg_right, WINDOW_DARK)

# 10 columns, each exactly 1 pixel wide, evenly spaced
n_cols = 10
for i in range(n_cols):
    cx = bldg_left + 1 + i * 2
    if cx >= bldg_right:
        break
    for r in range(15, 25):
        sp(r, cx, (3, 3, 3))  # white column

# cornice at top of columns
for c in range(bldg_left, bldg_right):
    sp(15, c, (2, 2, 2))

# flanking stone walls
fill_rect(15, 25, bldg_left, bldg_left + 1, (2, 2, 2))
fill_rect(15, 25, bldg_right - 1, bldg_right, (2, 2, 2))


# base / steps (rows 25-26)
fill_rect(25, 27, bldg_left - 1, bldg_right + 1, (1, 1, 1))


# trees (flanking the building)

# left tree: large, cols 0-5, canopy rows 6-18, trunk 19-26
# triangular canopy getting wider downward
for r in range(6, 19):
    half_w = min((r - 5) // 2, 3)
    for dc in range(-half_w, half_w + 1):
        c = 3 + dc
        color = TREE_LIGHT if dc == 0 or r % 2 == 0 else TREE_DARK
        sp(r, c, color)
# trunk
for r in range(19, 27):
    sp(r, 3, TRUNK)

# right tree: large, cols 27-31
for r in range(7, 20):
    half_w = min((r - 6) // 2, 3)
    for dc in range(-half_w, half_w + 1):
        c = 29 + dc
        color = TREE_LIGHT if dc == 0 or r % 2 == 0 else TREE_DARK
        sp(r, c, color)
for r in range(20, 27):
    sp(r, 29, TRUNK)

# small trees in front, left side
for r in range(18, 24):
    half_w = min((r - 17) // 2, 2)
    for dc in range(-half_w, half_w + 1):
        c = 1 + dc
        sp(r, c, TREE_LIGHT if r % 2 == 0 else TREE_DARK)
for r in range(24, 27):
    sp(r, 1, TRUNK)

# small tree front right
for r in range(19, 24):
    half_w = min((r - 18) // 2, 1)
    for dc in range(-half_w, half_w + 1):
        c = 31 + dc
        if 0 <= c < W:
            sp(r, c, TREE_DARK)
for r in range(24, 27):
    sp(r, 31, TRUNK)


# killian court lawn (rows 27-31)
for r in range(27, 32):
    for c in range(W):
        # slight variation
        sp(r, c, GRASS if (r + c) % 3 != 0 else GRASS_DARK)


# final pass: replace pure white background with faint cyan
# columns (rows 15-24) stay white; everything else that's (0,0,0) becomes faint cyan
for r in range(H):
    for c in range(W):
        if tuple(img[r, c]) == (0, 0, 0):
            # keep columns white (rows 15-24, at column positions)
            is_column = False
            if 15 <= r <= 24:
                for cx in [bldg_left + 1 + i * 2 for i in range(10)]:
                    if c == cx and cx < bldg_right:
                        is_column = True
                        break
            # keep clouds white (rows 0-8, already set as CLOUD)
            is_cloud = r <= 8
            if not is_column and not is_cloud:
                sp(r, c, FAINT_CYAN)


# output

np.save(os.path.join(DATA_DIR, 'mit_dome_pixels.npy'), img)

def cmy_to_rgb(arr):
    rgb = np.zeros_like(arr, dtype=np.uint8)
    rgb[:, :, 0] = np.clip(255 - arr[:, :, 0].astype(np.int32) * 85, 0, 255)
    rgb[:, :, 1] = np.clip(255 - arr[:, :, 1].astype(np.int32) * 85, 0, 255)
    rgb[:, :, 2] = np.clip(255 - arr[:, :, 2].astype(np.int32) * 85, 0, 255)
    return rgb

rgb = cmy_to_rgb(img)

fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=100)
ax.imshow(rgb, interpolation='nearest', aspect='equal')
ax.axis('off')
fig.tight_layout(pad=0)
fig.savefig(os.path.join(DATA_DIR, 'mit_dome_native.pdf'), dpi=300, bbox_inches='tight', pad_inches=0)
plt.close(fig)
print('saved mit_dome_native.pdf')

fig, ax = plt.subplots(figsize=(W * 15 / 72, H * 15 / 72), dpi=72)
ax.imshow(rgb, interpolation='nearest', aspect='equal', extent=[-0.5, W - 0.5, H - 0.5, -0.5])
for x in range(W + 1): ax.axvline(x - 0.5, color='gray', lw=0.3, alpha=0.5)
for y in range(H + 1): ax.axhline(y - 0.5, color='gray', lw=0.3, alpha=0.5)
ax.axis('off')
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(DATA_DIR, 'mit_dome_grid.pdf'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved mit_dome_grid.pdf')

total = W * H
unique = len(set(tuple(img[r, c]) for r in range(H) for c in range(W)))
print(f'\n{W}x{H} = {total} pixels, {unique}/64 unique colors')
all_ok = True
for ch, name in enumerate(['C', 'M', 'Y']):
    levels = []
    for lv in range(4):
        n = int(np.sum(img[:, :, ch] == lv))
        if n > 0:
            levels.append(lv)
            print(f'  {name}={lv}: {n} ({100 * n / total:.1f}%)')
    if len(levels) < 4:
        print(f'  WARNING: {name} missing levels {set(range(4)) - set(levels)}')
        all_ok = False
    else:
        print(f'  {name}: all 4 levels exercised')
if all_ok:
    print('\nall 4 levels exercised in all 3 channels')
