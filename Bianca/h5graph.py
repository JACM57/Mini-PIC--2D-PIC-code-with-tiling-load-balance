import h5py
import numpy as np
import matplotlib.pyplot as plt
import glob

# Define global grid size
tiles_per_row = 6
tile_size = 30
global_size = tiles_per_row * tile_size  # 180x180

# Placeholder for the reconstructed Ex field
Ex_full = np.zeros((global_size, global_size))

# Load all HDF5 files for a specific timestep
timestep = 0  # Change as needed
file_pattern = f"Simulation/Fields/fields_rank_*_step_{timestep}.h5"
files = sorted(glob.glob(file_pattern))  # Get all files for this timestep

for filename in files:
    with h5py.File(filename, "r") as f:
        for tile_name in f.keys():  # Loop through Tile_0, Tile_1, etc.
            if "Tile_" in tile_name:
                tile_id = int(tile_name.split("_")[1])  # Extract tile number
                tile_data = f[f"{tile_name}/fields"]["Ex"][:]  # Read Ex field

                # Compute global position
                row, col = divmod(tile_id, tiles_per_row)
                i_start, j_start = row * tile_size, col * tile_size

                # Place tile data in the correct position
                Ex_full[i_start:i_start+tile_size, j_start:j_start+tile_size] = tile_data

# Plot the reconstructed Ex field
plt.figure(figsize=(8, 6))
plt.imshow(Ex_full, cmap="inferno", origin="lower")
plt.colorbar(label="Ex field")
plt.title(f"Ex Field at timestep {timestep}")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
