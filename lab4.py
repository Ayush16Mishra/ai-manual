"""
jigsaw_sa.py
Jigsaw puzzle solver using Simulated Annealing (tile-swapping).

Usage:
    python jigsaw_sa.py           # uses synthetic image
    python jigsaw_sa.py path/to/image.png

Dependencies:
    pip install numpy pillow matplotlib
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import math
from copy import deepcopy

# -------------------------
# Utilities: image / tiles
# -------------------------


def load_image(path=None, size=(256,256)):
    """Load image from path, convert to RGB and resize. If path=None, generate gradient image."""
    if path is None:
        # generate synthetic gradient
        w, h = size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        xv = np.linspace(0, 255, w).astype(np.uint8)
        yv = np.linspace(0, 255, h).astype(np.uint8)
        gx = np.tile(xv, (h,1))
        gy = np.tile(yv[:,None], (1,w))
        img[...,0] = gx
        img[...,1] = gy
        img[...,2] = (gx // 2 + gy // 2)
        return Image.fromarray(img)
    else:
        img = Image.open(path).convert('RGB')
        img = img.resize(size, Image.BICUBIC)
        return img

def split_into_tiles(img, grid=(4,4)):
    """Split PIL image into tiles. Returns list of tiles (numpy arrays) and (rows,cols)."""
    w, h = img.size
    rows, cols = grid
    tw, th = w // cols, h // rows
    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tw
            upper = r * th
            tile = np.array(img.crop((left, upper, left+tw, upper+th)), dtype=np.float32)
            tiles.append(tile)
    return tiles, (rows, cols)

def assemble_from_permutation(tiles, perm, grid):
    """Assemble full image from tiles in order given by perm (list of indices)."""
    rows, cols = grid
    th, tw, _ = tiles[0].shape
    out = np.zeros((rows*th, cols*tw, 3), dtype=np.uint8)
    for pos_idx, tile_idx in enumerate(perm):
        r = pos_idx // cols
        c = pos_idx % cols
        out[r*th:(r+1)*th, c*tw:(c+1)*tw] = np.clip(tiles[tile_idx], 0, 255).astype(np.uint8)
    return out

# -------------------------
# Energy function
# -------------------------

def border_diff(a_edge, b_edge):
    """Compute dissimilarity between two 1D (edge) arrays of shape (N,3). Return scalar."""
    # L2 norm per pixel, then sum
    return np.sum(np.sqrt(np.sum((a_edge - b_edge)**2, axis=1)))

def precompute_edges(tiles):
    """Return dict of edges for each tile: top, bottom, left, right arrays."""
    edges = []
    for t in tiles:
        top = t[0, :, :]        # first row
        bottom = t[-1, :, :]    # last row
        left = t[:, 0, :]       # first column
        right = t[:, -1, :]     # last column
        edges.append({'top': top, 'bottom': bottom, 'left': left, 'right': right})
    return edges

def energy_of_permutation(perm, edges, grid):
    """Compute total energy (sum of border mismatches) for the given permutation mapping tiles->positions."""
    rows, cols = grid
    total = 0.0
    # For each position, check right neighbor and bottom neighbor
    for pos_idx, tile_idx in enumerate(perm):
        r = pos_idx // cols
        c = pos_idx % cols
        e = edges[tile_idx]
        # right neighbor
        if c < cols - 1:
            right_pos = pos_idx + 1
            right_tile_idx = perm[right_pos]
            e_right = edges[right_tile_idx]
            # compare right edge of current to left edge of right neighbor
            total += border_diff(e['right'], e_right['left'])
        # bottom neighbor
        if r < rows - 1:
            bottom_pos = pos_idx + cols
            bottom_tile_idx = perm[bottom_pos]
            e_bottom = edges[bottom_tile_idx]
            # compare bottom edge of current to top edge of bottom neighbor
            total += border_diff(e['bottom'], e_bottom['top'])
    return total

# -------------------------
# Simulated Annealing
# -------------------------

def random_swap_perm(perm):
    """Return a new permutation with two positions swapped."""
    new_perm = perm.copy()
    i, j = random.sample(range(len(perm)), 2)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm, (i, j)

def simulated_annealing(tiles, grid, T0=1e4, alpha=0.995, Tmin=1e-3, max_iters=200000, report_every=5000):
    """Run simulated annealing and return best permutation and energy trace."""
    n = len(tiles)
    # initial permutation: random
    current_perm = list(range(n))
    random.shuffle(current_perm)
    edges = precompute_edges(tiles)
    current_energy = energy_of_permutation(current_perm, edges, grid)
    best_perm = current_perm.copy()
    best_energy = current_energy
    energies = [current_energy]
    T = T0
    it = 0
    while T > Tmin and it < max_iters:
        new_perm, swapped = random_swap_perm(current_perm)
        new_energy = energy_of_permutation(new_perm, edges, grid)
        delta = new_energy - current_energy
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_perm = new_perm
            current_energy = new_energy
            if new_energy < best_energy:
                best_energy = new_energy
                best_perm = new_perm.copy()
        energies.append(current_energy)
        T *= alpha
        it += 1
        if it % report_every == 0:
            print(f"Iter {it:6d}  T={T:.5f}  current_energy={current_energy:.2f}  best_energy={best_energy:.2f}")
    print(f"Finished: it={it}, best_energy={best_energy:.2f}, current_energy={current_energy:.2f}")
    return best_perm, best_energy, energies

# -------------------------
# Helper: scramble, show
# -------------------------

def make_scrambled(tiles, grid, seed=0):
    """Return a scrambled permutation (random) and the scrambled image."""
    n = len(tiles)
    perm = list(range(n))
    random.seed(seed)
    random.shuffle(perm)
    scrambled_img = assemble_from_permutation(tiles, perm, grid)
    return perm, scrambled_img

def plot_results(original_img, scrambled_img, solved_img, energies):
    fig, axs = plt.subplots(1,3, figsize=(15,5))
    axs[0].imshow(original_img); axs[0].set_title("Original (ground-truth)"); axs[0].axis('off')
    axs[1].imshow(scrambled_img); axs[1].set_title("Scrambled (initial)"); axs[1].axis('off')
    axs[2].imshow(solved_img); axs[2].set_title("Solved (SA result)"); axs[2].axis('off')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(energies)
    plt.yscale('log')
    plt.xlabel("Iteration")
    plt.ylabel("Energy (log scale)")
    plt.title("Energy vs Iterations")
    plt.grid(True)
    plt.show()

# -------------------------
# Main runnable example
# -------------------------

def main(image_path=None):
    # Parameters
    image_size = (256, 256)   # (width, height)
    grid = (4,4)              # 4x4 puzzle => 16 tiles
    T0 = 1e4
    alpha = 0.995
    Tmin = 1e-3
    max_iters = 50000
    report_every = 5000
    seed = 42
    if image_path is None:
            image_path = "C:\\Users\\ashmi\\OneDrive\\Desktop\\manual\\lab4\\test.jpg"
    # Load or generate image
    pil_img = load_image(image_path, size=image_size)
    original_np = np.array(pil_img)

    # Split into tiles
    tiles, grid_sz = split_into_tiles(pil_img, grid=grid)
    assert grid_sz == grid

    # Keep the true "ordered" perm as ground-truth assembly
    true_perm = list(range(len(tiles)))
    true_img = assemble_from_permutation(tiles, true_perm, grid)

    # Make scrambled
    random.seed(seed)
    scrambled_perm, scrambled_img = make_scrambled(tiles, grid, seed=seed)

    plt.figure(figsize=(6,6))
    plt.imshow(scrambled_img); plt.title("Initial scrambled"); plt.axis('off')
    plt.show()

    # Run simulated annealing
    best_perm, best_energy, energies = simulated_annealing(tiles, grid, T0=T0, alpha=alpha, Tmin=Tmin, max_iters=max_iters, report_every=report_every)

    # Reconstruct images
    solved_img = assemble_from_permutation(tiles, best_perm, grid)

    # Plot results
    plot_results(true_img, scrambled_img, solved_img, energies)

    # Optionally save images
    Image.fromarray(solved_img).save("solved_result.png")
    print("Solved image saved to solved_result.png")

if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    main(img_path)
