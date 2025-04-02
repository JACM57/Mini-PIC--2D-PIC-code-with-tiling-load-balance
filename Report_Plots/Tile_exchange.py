import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Grid configuration
cols = 6
rows = 6
tile_total = cols * rows
rank_tile_size = 3  # Each rank block is 3x3 tiles

# Define colors for the 4 ranks (2x2 grid of rank blocks)
rank_colors = {
    0: 'lightblue',    # Top-left rank (rank 0)
    1: 'lightgreen',   # Top-right rank (rank 1)
    2: 'lightyellow',  # Bottom-left rank (rank 2)
    3: 'lightcoral'    # Bottom-right rank (rank 3)
}

def get_rank_id(tile_row, tile_col, rank_tile_size, cols):
    """
    Compute the rank ID for a tile given its (tile_row, tile_col).
    We assume a row-major ordering of rank blocks.
    """
    rank_row = tile_row // rank_tile_size
    rank_col = tile_col // rank_tile_size
    # Number of rank columns = cols // rank_tile_size
    rank_id = rank_row * (cols // rank_tile_size) + rank_col
    return rank_id

# ---------------------
# Image 1: Each tile colored by its rank.
# ---------------------
fig1, ax1 = plt.subplots(figsize=(6, 6))

for i in range(tile_total):
    tile_row = i // cols  # logical row (0 = top row)
    tile_col = i % cols   # logical column
    rank_id = get_rank_id(tile_row, tile_col, rank_tile_size, cols)
    color = rank_colors[rank_id]
    
    # Compute drawing coordinates (flip y so that tile 0 is top-left)
    x = tile_col
    y = rows - tile_row - 1
    rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='none')
    ax1.add_patch(rect)

# Draw black tile grid lines (zorder 2 so they appear over the tile colors)
for x in range(cols + 1):
    ax1.plot([x, x], [0, rows], color='black', linewidth=2, zorder=2)
for y in range(rows + 1):
    ax1.plot([0, cols], [y, y], color='black', linewidth=2, zorder=2)

# Draw internal red rank grid lines (every 3 tiles, zorder 3)
for x in range(rank_tile_size, cols, rank_tile_size):
    ax1.plot([x, x], [0, rows], color='red', linewidth=4, zorder=3)
for y in range(rank_tile_size, rows, rank_tile_size):
    ax1.plot([0, cols], [y, y], color='red', linewidth=4, zorder=3)

# Draw outer red border explicitly (zorder 4)
outer_border = patches.Rectangle((0, 0), cols, rows, linewidth=4, edgecolor='red', facecolor='none', zorder=4)
ax1.add_patch(outer_border)

# Add tile numbers (zorder 5)
for i in range(tile_total):
    tile_row = i // cols
    tile_col = i % cols
    x = tile_col
    y = rows - tile_row - 1
    ax1.text(x + 0.5, y + 0.5, str(i), ha='center', va='center', fontsize=14, color='black', zorder=5)

ax1.set_xlim(-0.1, cols + 0.1)
ax1.set_ylim(-0.1, rows + 0.1)
ax1.set_aspect('equal')
ax1.axis('off')
plt.tight_layout()
plt.savefig("Report_Plots/Image1_RankColor.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()

# ---------------------
# Image 2: Override tiles 6, 9, and 24 with the bottom-right rank's color.
# ---------------------
# Determine the bottom-right rank:
# The bottom-right tile is at (tile_row, tile_col) = (rows-1, cols-1)
bottom_right_rank = get_rank_id(rows - 1, cols - 1, rank_tile_size, cols)
br_color = rank_colors[bottom_right_rank]

# Tiles to override:
override_tiles = {6, 9, 24}

fig2, ax2 = plt.subplots(figsize=(6, 6))

for i in range(tile_total):
    tile_row = i // cols
    tile_col = i % cols
    rank_id = get_rank_id(tile_row, tile_col, rank_tile_size, cols)
    color = rank_colors[rank_id]
    
    # Override specified tiles with bottom-right rank's color.
    if i in override_tiles:
        color = br_color

    x = tile_col
    y = rows - tile_row - 1
    rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='none')
    ax2.add_patch(rect)

# Draw grid lines and red rank boundaries as in image 1:
for x in range(cols + 1):
    ax2.plot([x, x], [0, rows], color='black', linewidth=2, zorder=2)
for y in range(rows + 1):
    ax2.plot([0, cols], [y, y], color='black', linewidth=2, zorder=2)
for x in range(rank_tile_size, cols, rank_tile_size):
    ax2.plot([x, x], [0, rows], color='red', linewidth=4, zorder=3)
for y in range(rank_tile_size, rows, rank_tile_size):
    ax2.plot([0, cols], [y, y], color='red', linewidth=4, zorder=3)
outer_border = patches.Rectangle((0, 0), cols, rows, linewidth=4, edgecolor='red', facecolor='none', zorder=4)
ax2.add_patch(outer_border)
for i in range(tile_total):
    tile_row = i // cols
    tile_col = i % cols
    x = tile_col
    y = rows - tile_row - 1
    ax2.text(x + 0.5, y + 0.5, str(i), ha='center', va='center', fontsize=14, color='black', zorder=5)

ax2.set_xlim(-0.1, cols + 0.1)
ax2.set_ylim(-0.1, rows + 0.1)
ax2.set_aspect('equal')
ax2.axis('off')
plt.tight_layout()
plt.savefig("Report_Plots/Image2_Override.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()
