import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Grid configuration
cols = 6
rows = 6
tile_total = cols * rows
rank_tile_size = 3  # Each rank block is 3x3 tiles

# Compute neighbor indices for tile 0 using periodic wrapping.
neighbors = set()
r0, c0 = 0, 0
for dr in [-1, 0, 1]:
    for dc in [-1, 0, 1]:
        if dr == 0 and dc == 0:
            continue
        nr = (r0 + dr) % rows  # wrap rows
        nc = (c0 + dc) % cols  # wrap columns
        neighbors.add(nr * cols + nc)
# Expected neighbor set: {35, 30, 31, 5, 1, 11, 6, 7}

fig, ax = plt.subplots(figsize=(6, 6))

# --- Draw colored tiles ---
for i in range(tile_total):
    r = i // cols
    c = i % cols
    x = c
    y = rows - r - 1
    if i == 0:
        color = 'blue'     # Tile 0 in blue
    elif i in neighbors:
        color = 'green'    # Neighbors in green
    else:
        color = 'white'    # Others in white
    rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='none', zorder=1)
    ax.add_patch(rect)

# --- Draw black tile grid lines (zorder=2 so they're above tiles, but below red lines) ---
for x in range(cols + 1):
    ax.plot([x, x], [0, rows], color='black', linewidth=2, zorder=2)
for y in range(rows + 1):
    ax.plot([0, cols], [y, y], color='black', linewidth=2, zorder=2)

# --- Draw internal red rank grid lines (zorder=3) ---
for x in range(rank_tile_size, cols, rank_tile_size):
    ax.plot([x, x], [0, rows], color='red', linewidth=4, zorder=3)
for y in range(rank_tile_size, rows, rank_tile_size):
    ax.plot([0, cols], [y, y], color='red', linewidth=4, zorder=3)

# --- Draw the outer red border (zorder=4 to ensure it's on top) ---
outer_border = patches.Rectangle(
    (0, 0), cols, rows,
    linewidth=4, edgecolor='red', facecolor='none', zorder=4
)
ax.add_patch(outer_border)

# --- Add tile numbers (zorder=5 so they remain fully visible) ---
for i in range(tile_total):
    r = i // cols
    c = i % cols
    x = c
    y = rows - r - 1
    ax.text(x + 0.5, y + 0.5, str(i), ha='center', va='center',
            fontsize=14, color='black', zorder=5)

# --- Final plot settings ---
ax.set_xlim(-0.1, cols + 0.1)
ax.set_ylim(-0.1, rows + 0.1)
ax.set_aspect('equal')
ax.axis('off')

# Save with minimal padding
plt.savefig("Report_Plots/GID.pdf", bbox_inches='tight', pad_inches=0.05)
plt.show()




##############################################

import matplotlib.pyplot as plt

# Grid configuration
cols = 6
rows = 6
tile_total = cols * rows
rank_tile_size = 3  # Each rank has a 3x3 tile block

fig, ax = plt.subplots(figsize=(6, 6))

# --- Draw black tile grid ---
for x in range(cols + 1):
    ax.plot([x, x], [0, rows], color='black', linewidth=2)
for y in range(rows + 1):
    ax.plot([0, cols], [y, y], color='black', linewidth=2)

# --- Draw red rank grid (including outer boundaries) ---
rank_lines_x = list(range(0, cols + 1, rank_tile_size))
rank_lines_y = list(range(0, rows + 1, rank_tile_size))

for x in rank_lines_x:
    ax.plot([x, x], [0, rows], color='red', linewidth=4)
for y in rank_lines_y:
    ax.plot([0, cols], [y, y], color='red', linewidth=4)

# --- Number each tile in row-major order ---
for i in range(tile_total):
    row = i // cols
    col = i % cols
    x = col
    y = rows - row - 1  # Flip vertically to match top-left origin
    ax.text(x + 0.5, y + 0.5, str(i), ha='center', va='center', fontsize=14)

# Final plot settings
ax.set_xlim(-0.2, cols + 0.2)
ax.set_ylim(-0.2, rows + 0.2)
ax.set_aspect('equal')
ax.axis('off')
plt.tight_layout()
#plt.savefig("Report_Plots/GID.pdf")
plt.show()



