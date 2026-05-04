#!/usr/bin/env python3
"""
5-frame animation of Tim the Beaver walking across the MIT dome scene.
each frame is 32x32 CMY, 4 intensity levels per channel.
the background is the MIT dome; the beaver walks left to right across the lawn.
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
N_FRAMES = 5

# load the MIT dome background
bg = np.load(os.path.join(DATA_DIR, 'mit_dome_pixels.npy')).copy()


def cmy_to_rgb(arr):
    rgb = np.zeros_like(arr, dtype=np.uint8)
    rgb[:, :, 0] = np.clip(255 - arr[:, :, 0].astype(np.int32) * 85, 0, 255)
    rgb[:, :, 1] = np.clip(255 - arr[:, :, 1].astype(np.int32) * 85, 0, 255)
    rgb[:, :, 2] = np.clip(255 - arr[:, :, 2].astype(np.int32) * 85, 0, 255)
    return rgb


def sp(img, r, c, cmy):
    if 0 <= r < H and 0 <= c < W:
        img[r, c] = cmy


# beaver sprite: ~8 wide x 6 tall pixel art
# drawn relative to (feet_row, left_col)
# CMY colors:
BROWN_DARK = (1, 2, 2)   # dark brown body
BROWN_LIGHT = (0, 2, 1)  # lighter brown belly
BROWN_MED = (1, 2, 1)    # medium brown
TAIL_DARK = (1, 3, 2)    # dark flat tail
EYE = (0, 0, 0)          # white eye
NOSE = (0, 3, 3)         # black nose
TEETH = (0, 0, 0)        # white teeth


def draw_beaver(img, feet_row, left_col, leg_phase):
    """draw a beaver sprite. leg_phase: 0-4 controls leg animation."""
    fr = feet_row
    lc = left_col

    # tail (flat, horizontal, behind body) - row fr-2 to fr-1
    for dc in range(-3, 0):
        sp(img, fr - 2, lc + dc, TAIL_DARK)
        sp(img, fr - 1, lc + dc, TAIL_DARK)

    # body (oval, 5 wide x 4 tall)
    # row fr-5: top of head
    sp(img, fr - 5, lc + 2, BROWN_DARK)
    sp(img, fr - 5, lc + 3, BROWN_DARK)
    sp(img, fr - 5, lc + 4, BROWN_DARK)

    # row fr-4: head
    for dc in range(1, 6):
        sp(img, fr - 4, lc + dc, BROWN_DARK)

    # row fr-3: head with eye and nose
    sp(img, fr - 3, lc + 0, BROWN_DARK)
    sp(img, fr - 3, lc + 1, BROWN_DARK)
    sp(img, fr - 3, lc + 2, BROWN_DARK)
    sp(img, fr - 3, lc + 3, BROWN_LIGHT)  # belly starts
    sp(img, fr - 3, lc + 4, BROWN_DARK)
    sp(img, fr - 3, lc + 5, BROWN_DARK)
    sp(img, fr - 3, lc + 6, NOSE)  # nose at front

    # eye
    sp(img, fr - 4, lc + 5, EYE)

    # teeth (below nose)
    sp(img, fr - 2, lc + 6, TEETH)

    # row fr-2: body
    sp(img, fr - 2, lc + 0, BROWN_DARK)
    sp(img, fr - 2, lc + 1, BROWN_DARK)
    sp(img, fr - 2, lc + 2, BROWN_LIGHT)
    sp(img, fr - 2, lc + 3, BROWN_LIGHT)
    sp(img, fr - 2, lc + 4, BROWN_DARK)
    sp(img, fr - 2, lc + 5, BROWN_DARK)

    # row fr-1: lower body
    sp(img, fr - 1, lc + 0, BROWN_DARK)
    sp(img, fr - 1, lc + 1, BROWN_LIGHT)
    sp(img, fr - 1, lc + 2, BROWN_LIGHT)
    sp(img, fr - 1, lc + 3, BROWN_LIGHT)
    sp(img, fr - 1, lc + 4, BROWN_DARK)
    sp(img, fr - 1, lc + 5, BROWN_DARK)

    # legs (animated)
    # 4 legs: 2 front, 2 back
    # phase determines which pair is forward
    if leg_phase % 2 == 0:
        # back legs spread, front legs together
        sp(img, fr, lc + 1, BROWN_DARK)  # back left
        sp(img, fr, lc + 2, BROWN_DARK)  # back right (spread)
        sp(img, fr, lc + 4, BROWN_DARK)  # front (together)
    else:
        # back legs together, front legs spread
        sp(img, fr, lc + 1, BROWN_DARK)  # back (together)
        sp(img, fr, lc + 3, BROWN_DARK)  # front left
        sp(img, fr, lc + 5, BROWN_DARK)  # front right (spread)


# generate 5 frames with beaver walking left to right
# beaver walks across the lawn (row 31 = feet)
beaver_row = 30  # feet row
x_positions = [2, 8, 14, 20, 26]  # left edge of beaver, walking right

frames = []
for frame_idx in range(N_FRAMES):
    img = bg.copy()
    bx = x_positions[frame_idx]
    draw_beaver(img, beaver_row, bx, frame_idx)
    frames.append(img)

    # save individual frame
    np.save(os.path.join(DATA_DIR, f'beaver_frame_{frame_idx}.npy'), img)

    # render
    rgb = cmy_to_rgb(img)
    fig, ax = plt.subplots(figsize=(W * 15 / 72, H * 15 / 72), dpi=72)
    ax.imshow(rgb, interpolation='nearest', aspect='equal',
              extent=[-0.5, W - 0.5, H - 0.5, -0.5])
    for x in range(W + 1):
        ax.axvline(x - 0.5, color='gray', lw=0.3, alpha=0.5)
    for y in range(H + 1):
        ax.axhline(y - 0.5, color='gray', lw=0.3, alpha=0.5)
    ax.set_title(f'Frame {frame_idx}', fontsize=8)
    ax.axis('off')
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(DATA_DIR, f'beaver_frame_{frame_idx}.pdf'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)

# render all 5 frames in a strip
fig, axes = plt.subplots(1, 5, figsize=(15, 3.2))
for i in range(5):
    rgb = cmy_to_rgb(frames[i])
    axes[i].imshow(rgb, interpolation='nearest', aspect='equal')
    axes[i].set_title(f'Frame {i}', fontsize=8)
    axes[i].axis('off')
fig.suptitle('Walking beaver animation (5 frames)', fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(DATA_DIR, 'beaver_animation_strip.pdf'), dpi=150, bbox_inches='tight')
plt.close(fig)
print('saved beaver_animation_strip.pdf')

# also save as animated GIF
from PIL import Image
gif_frames = []
for frame in frames:
    rgb = cmy_to_rgb(frame)
    # scale up 10x for visibility
    pil_img = Image.fromarray(rgb).resize((320, 320), Image.NEAREST)
    gif_frames.append(pil_img)

gif_frames[0].save(os.path.join(DATA_DIR, 'beaver_walk.gif'),
                    save_all=True, append_images=gif_frames[1:],
                    duration=300, loop=0)
print('saved beaver_walk.gif')

# stats
total_pixels = N_FRAMES * W * H
print(f'\n{N_FRAMES} frames x {W}x{H} = {total_pixels} pixels')
print(f'total oligos needed: {total_pixels} (1 per pixel)')
all_frames = np.stack(frames)
unique = len(set(tuple(all_frames[f, r, c]) for f in range(N_FRAMES)
                 for r in range(H) for c in range(W)))
print(f'unique colors across all frames: {unique}/64')
for ch, name in enumerate(['C', 'M', 'Y']):
    levels = set()
    for lv in range(4):
        if np.any(all_frames[:, :, :, ch] == lv):
            levels.add(lv)
    print(f'  {name} levels: {sorted(levels)}')
