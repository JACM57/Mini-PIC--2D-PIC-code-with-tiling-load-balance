#!/usr/bin/env python3

import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt

def load_field(step,
                  folder="Simulation/Fields",
                  quantity="Ex",
                  box_x=10.0,
                  box_y=10.0,
                  nx_global=180,
                  ny_global=180,
                  guard=2,
                  interior_nx=30,
                  interior_ny=30):
    """
    Reads all HDF5 files for the given time step (fields_rank_*_step_{step}.h5)
    and reconstructs the chosen field component ('Ex', 'Ey', 'Ez', 'Bx', 'By', 'Bz')
    over the entire domain of size (nx_global, ny_global).
    
    Assumptions:
      - Each file has groups named '/Tile_<ID>', each with:
          dataset "fields" of shape [ny_tile, nx_tile] = [interior_ny+2*g, interior_nx+2*g]
          attributes: tileRow, tileCol, currentRank
      - The tile’s interior is from index guard..(guard+interior_nx-1) in x
        and guard..(guard+interior_ny-1) in y.
      - tileRow=0 => bottom row, tileCol=0 => left column
      - The entire domain is (nx_global x ny_global) cells, physically spanning [0, box_x] x [0, box_y].
      - box_x / nx_global => dx, box_y / ny_global => dy.
    """

    # Initialize the global field array
    field_global = np.zeros((ny_global, nx_global), dtype=np.float64)

    # Collect all HDF5 files for this step
    pattern = f"{folder}/fields_rank_*_step_{step}.h5"
    file_list = glob.glob(pattern)

    # Read each rank file
    for filename in file_list:
        with h5py.File(filename, "r") as f:
            # Loop over each group named "Tile_<GID>"
            for group_name in f.keys():
                if not group_name.startswith("Tile_"):
                    continue
                tile_group = f[group_name]
                
                # Attributes telling us where this tile belongs in the global layout
                tile_row = tile_group.attrs["tileRow"]
                tile_col = tile_group.attrs["tileCol"]
                currentRank = tile_group.attrs["currentRank"]

                
                # Read the entire 2D dataset (including guard)
                # shape = (interior_ny + 2*guard, interior_nx + 2*guard)
                dataset = tile_group["fields"][:]
                ny_tile, nx_tile = dataset.shape  # each element is a struct (Ex,Ey,Ez,Bx,By,Bz)
                
                # Verify that matches what we expect
                # (not strictly necessary, but good practice)
                if ny_tile != (interior_ny + 2*guard) or nx_tile != (interior_nx + 2*guard):
                    raise ValueError(
                        f"Inconsistent tile shape in {filename}/{group_name}: "
                        f"got {ny_tile}x{nx_tile}, expected {(interior_ny + 2*guard)}x{(interior_nx + 2*guard)}"
                    )
                
                # The tile's interior portion is [guard : guard+interior_ny] in Y,
                # and [guard : guard+interior_nx] in X.
                # Place that portion into the global array at the correct offset.
                # If the tile covers global domain rows [tile_row*interior_ny : (tile_row+1)*interior_ny),
                # and columns [tile_col*interior_nx : (tile_col+1)*interior_nx),
                # then the offset in the global array is:
                globalTileRows = ny_global // interior_ny

                #row_offset = (globalTileRows - 1 - tile_row) * interior_ny
                row_offset = tile_row * interior_ny
                col_offset = tile_col * interior_nx
                
                for j in range(interior_ny):
                    for i in range(interior_nx):
                        # Extract the desired field component from the compound dataset
                        # Each dataset[j,i] is a struct with named fields
                        # In numpy, we'd do: dataset[j,i]["Ex"] if it's a compound type
                        field_val = dataset[j + guard, i + guard][quantity]
                        
                        # Place it in the global array
                        field_global[row_offset + j, col_offset + i] = field_val
    
    return field_global


def plot_field(field_array, box_x=10.0, box_y=10.0, quantity="Ex"):
    """
    Given a 2D NumPy array of shape (ny_global, nx_global) and the physical domain
    [0..box_x] x [0..box_y], plot it with pcolormesh, using origin='lower'.
    """
    ny_global, nx_global = field_array.shape
    dx = box_x / nx_global
    dy = box_y / ny_global
    
    # Create coordinates for cell edges
    x_edges = np.linspace(0, box_x, nx_global + 1)
    y_edges = np.linspace(0, box_y, ny_global + 1)
    
    plt.figure()
    plt.pcolormesh(x_edges, y_edges, field_array, shading='auto')

    plt.colorbar(label=quantity)
    plt.xlabel("$x \\,[c/\\omega_p]$")
    plt.ylabel("$y \\,[c/\\omega_p]$")
    plt.title(f"{quantity} field [Simulation Units]")
    plt.gca().set_aspect("equal", "box")
    plt.show()


if __name__ == "__main__":
    # Example usage:
    # 1) Read step=0 from the default "Simulation/Fields/" folder
    # 2) Reconstruct Ex (or any other quantity you want) over the entire domain of 180x180 cells
    # 3) Plot it
    step_to_load = 0

    ex_data = load_field(
        step=step_to_load,
        folder="Simulation/Fields",
        quantity="Ex",
        box_x=10.0,
        box_y=10.0,
        nx_global=180,
        ny_global=180,
        guard=2,
        interior_nx=30,
        interior_ny=30 
    )
    ey_data = load_field(
        step=step_to_load,
        folder="Simulation/Fields",
        quantity="Ey",
        box_x=10.0,
        box_y=10.0,
        nx_global=180,
        ny_global=180,
        guard=2,
        interior_nx=30,
        interior_ny=30 
    )
    ez_data = load_field(
        step=step_to_load,
        folder="Simulation/Fields",
        quantity="Ez",
        box_x=10.0,
        box_y=10.0,
        nx_global=180,
        ny_global=180,
        guard=2,
        interior_nx=30,
        interior_ny=30 
    )
    bx_data = load_field(
        step=step_to_load,
        folder="Simulation/Fields",
        quantity="Bx",
        box_x=10.0,
        box_y=10.0,
        nx_global=180,
        ny_global=180,
        guard=2,
        interior_nx=30,
        interior_ny=30 
    )
    by_data = load_field(
        step=step_to_load,
        folder="Simulation/Fields",
        quantity="By",
        box_x=10.0,
        box_y=10.0,
        nx_global=180,
        ny_global=180,
        guard=2,
        interior_nx=30,
        interior_ny=30 
    )
    bz_data = load_field(
        step=step_to_load,
        folder="Simulation/Fields",
        quantity="Bz",
        box_x=10.0,
        box_y=10.0,
        nx_global=180,
        ny_global=180,
        guard=2,
        interior_nx=30,
        interior_ny=30 
    )

    plot_field(ex_data, box_x=10.0, box_y=10.0, quantity="Ex")
    plot_field(ey_data, box_x=10.0, box_y=10.0, quantity="Ey")
    plot_field(ez_data, box_x=10.0, box_y=10.0, quantity="Ez")
    plot_field(bx_data, box_x=10.0, box_y=10.0, quantity="Bx")
    plot_field(by_data, box_x=10.0, box_y=10.0, quantity="By")
    plot_field(bz_data, box_x=10.0, box_y=10.0, quantity="Bz")