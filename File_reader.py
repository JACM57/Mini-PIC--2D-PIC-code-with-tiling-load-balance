#!/usr/bin/env python3

import os
import glob
import h5py
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation

##############################################################################
# 1) Read parameters from params.txt
##############################################################################

def read_params(param_file="Simulation/Fields/params.txt"):
    """
    Reads a simple text file with lines like:
        box_x=10.0
        nx_global=180
        ...
    Returns a dictionary: {"box_x": 10.0, "nx_global": 180, ...}
    """
    params = {}
    if not os.path.exists(param_file):
        print(f"WARNING: Parameter file '{param_file}' not found. Using defaults.")
        return params

    with open(param_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, val_str = line.split('=', 1)
            key = key.strip()
            val_str = val_str.strip()
            # Try to convert to float or int
            if '.' in val_str or 'e' in val_str.lower():
                # e.g. "1.0" or "1e-3"
                try:
                    params[key] = float(val_str)
                except ValueError:
                    params[key] = val_str  # fallback to string
            else:
                # try integer
                try:
                    params[key] = int(val_str)
                except ValueError:
                    params[key] = val_str
    return params

##############################################################################
# 2) The function to load HDF5 data into a global 2D array
##############################################################################

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
    into a 2D NumPy array of shape (ny_global, nx_global).

    - folder: the directory containing the HDF5 files
    - quantity: which field to extract ("Ex", "Ey", "Ez", "Bx", "By", "Bz")
    - box_x, box_y: physical size in x and y (used only for reference if you want)
    - nx_global, ny_global: total number of cells in x and y
    - guard: number of guard cells
    - interior_nx, interior_ny: interior cells per tile in x and y
    """

    field_global = np.zeros((ny_global, nx_global), dtype=np.float64)

    pattern = f"{folder}/fields_rank_*_step_{step}.h5"
    file_list = glob.glob(pattern)
    if not file_list:
        print(f"WARNING: No files found matching '{pattern}'.")
    
    for filename in file_list:
        with h5py.File(filename, "r") as f:
            for group_name in f.keys():
                if not group_name.startswith("Tile_"):
                    continue
                tile_group = f[group_name]
                
                tile_row = tile_group.attrs["tileRow"]
                tile_col = tile_group.attrs["tileCol"]
                # currentRank = tile_group.attrs["currentRank"]  # not strictly needed here

                dataset = tile_group["fields"][:]
                ny_tile, nx_tile = dataset.shape  # includes guard cells

                # Sanity check
                expected_y = interior_ny + 2*guard
                expected_x = interior_nx + 2*guard
                if ny_tile != expected_y or nx_tile != expected_x:
                    raise ValueError(
                        f"Inconsistent tile shape in {filename}/{group_name}: "
                        f"got {ny_tile}x{nx_tile}, expected {expected_y}x{expected_x}"
                    )

                # The interior portion is [guard:guard+interior_ny, guard:guard+interior_nx]
                row_offset = tile_row * interior_ny
                col_offset = tile_col * interior_nx

                for j in range(interior_ny):
                    for i in range(interior_nx):
                        field_val = dataset[j + guard, i + guard][quantity]
                        field_global[row_offset + j, col_offset + i] = field_val
    
    return field_global

##############################################################################
# 3) Function to plot the resulting 2D array
##############################################################################

def plot_field(field_array, box_x=10.0, box_y=10.0, quantity="Ex", dt=1.0, step=0):
    """
    Plots the 2D field array using pcolormesh with physical coordinates.
    """
    ny_global, nx_global = field_array.shape
    x_edges = np.linspace(0, box_x, nx_global + 1)
    y_edges = np.linspace(0, box_y, ny_global + 1)
    sim_time = step * dt

    plt.figure()
    #plt.pcolormesh(x_edges, y_edges, field_array, shading='auto', cmap="viridis")
    mesh = plt.pcolormesh(x_edges, y_edges, field_array, shading='auto', cmap="viridis")
    cbar = plt.colorbar(mesh)
    cbar.set_label(fr"{quantity} $[m_e c \omega_p / e]$", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.xlabel("$x \\,[c/\\omega_p]$", fontsize=14)
    plt.ylabel("$y \\,[c/\\omega_p]$", fontsize=14)
    plt.title(fr"{quantity} at $t = {sim_time:.3f} \, [\omega_p^{{-1}}]$", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_aspect("equal", "box")
    plt.savefig(f"Simulation_Videos/{quantity}_step_{step}.png")
    plt.show()

##############################################################################
# 4) Function to create a video for the evolution of the fields
##############################################################################

def create_field_animation(quantity="Ex", folder="Simulation/Fields", output_folder="Simulation_Videos", fps=20):
    params = read_params(os.path.join(folder, "params.txt"))
    box_x = params.get("box_x", 10.0)
    box_y = params.get("box_y", 10.0)
    nx_global = int(params.get("nx_global", 180))
    ny_global = int(params.get("ny_global", 180))
    guard = int(params.get("guard", 2))
    interior_nx = int(params.get("interior_nx", 30))
    interior_ny = int(params.get("interior_ny", 30))
    dt = params.get("dt", 1.0)

    os.makedirs(output_folder, exist_ok=True)
    out_file = os.path.join(output_folder, f"{quantity}_animation.mp4")

    step_files = sorted(glob.glob(os.path.join(folder, "fields_rank_0_step_*.h5")))
    if not step_files:
        raise RuntimeError(f"No HDF5 step files found in '{folder}' matching 'fields_rank_0_step_*.h5'")
    steps = sorted([int(f.split("_step_")[1].split(".h5")[0]) for f in step_files])

    frames = []
    for s in steps:
        field = load_field(s, folder, quantity, box_x, box_y,
                           nx_global, ny_global, guard, interior_nx, interior_ny)
        frames.append(field)

    fig, ax = plt.subplots()
    x_edges = np.linspace(0, box_x, nx_global + 1)
    y_edges = np.linspace(0, box_y, ny_global + 1)
    mesh = ax.pcolormesh(x_edges, y_edges, frames[0], shading='auto', cmap='viridis')
    cbar = fig.colorbar(mesh, ax=ax, label=fr"{quantity} $[m_e c \omega_p / e]$")
    title = ax.set_title("")
    ax.set_xlabel("$x \\,[c/\\omega_p]$")
    ax.set_ylabel("$y \\,[c/\\omega_p]$")
    ax.set_aspect('equal', 'box')

    def update(frame_idx):
        field_data = frames[frame_idx]
        mesh.set_array(field_data.ravel())
        #vmin = field_data.min()
        #vmax = field_data.max()
        #mesh.set_clim(vmin=vmin, vmax=vmax)
        #cbar.set_ticks([vmin, 0.0, vmax])
        cbar.update_normal(mesh)
        time = steps[frame_idx] * dt
        title.set_text(f"{quantity} at $t = {time:.3f} \\, [\omega_p^{{-1}}]$")
        return mesh, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames),
                                  interval=1000 // fps, blit=False, repeat=False)
    ani.save(out_file, writer="ffmpeg", dpi=150, fps=fps)
    plt.close(fig)
    print(f"Animation saved to {out_file}")

##############################################################################
# 5) Lineout plots
##############################################################################

def plot_line_slices_along_x_steps(
    steps_to_plot,
    quantity="Bz",
    folder="Simulation/Fields",
    y_index=None,
    output_file="line_slices_Bz.png"
):
    """
    Loads the field at the specified time steps and plots a horizontal lineout along x
    (at a fixed y-index) in a single figure.

    Parameters:
        steps_to_plot (list of int): List of simulation steps to plot.
        quantity (str): Field component to plot (default "Bz").
        folder (str): Directory containing the HDF5 files and params.txt.
        y_index (int): Row index (y) to slice along. If None, defaults to the middle row.
        output_file (str): Name of the output PNG file.
    """
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt

    # 1) Read simulation parameters.
    params = read_params(os.path.join(folder, "params.txt"))
    box_x = params.get("box_x", 10.0)
    box_y = params.get("box_y", 10.0)
    nx_global = int(params.get("nx_global", 180))
    ny_global = int(params.get("ny_global", 180))
    guard = int(params.get("guard", 2))
    interior_nx = int(params.get("interior_nx", 30))
    interior_ny = int(params.get("interior_ny", 30))
    dt = params.get("dt", 1.0)

    # 2) Determine which y-index to slice. Default to the middle row.
    if y_index is None:
        y_index = ny_global // 2

    # 3) Initialize list to hold the lineouts and corresponding times.
    line_data = []
    times = []
    for s in steps_to_plot:
        field_2d = load_field(s, folder, quantity, box_x, box_y,
                              nx_global, ny_global, guard,
                              interior_nx, interior_ny)
        # Extract horizontal slice at row y_index.
        lineout = field_2d[y_index, :]
        line_data.append(lineout)
        t_physical = s * dt
        times.append(t_physical)

    # 4) Create x-axis for the lineouts (assume uniform spacing).
    x_vals = np.linspace(0, box_x, nx_global)

    # 5) Plot all lineouts on the same figure.
    plt.figure(figsize=(10, 6))
    for idx, s in enumerate(steps_to_plot):
        label_str = f"Step {s} ($t = {times[idx]:.1f} \\, [\\omega_p^{{-1}}]$)"
        plt.plot(x_vals, line_data[idx], label=label_str)

    plt.title(f"{quantity} lineout along $x$ ($y = 5 \\, [c/\\omega_p]$) using nx = ny = {nx_global}", fontsize=18)
    plt.xlabel("$x \\, [c/\\omega_p]$", fontsize=18)
    plt.ylabel(f"${quantity} \\, [m_e \\, c \\, \\omega_p / e]$", fontsize=16)
    plt.xlim(0, box_x)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=12)
    plt.tight_layout()

    # 6) Save and display the figure.
    plt.savefig(output_file, dpi=150)
    plt.show()

    print(f"Line slices plot saved to '{output_file}'.")


##############################################################################
# 6) Error Evolution
##############################################################################

def track_peak_amplitudes_over_time(
    quantity="Bz",
    folder="Simulation/Fields",
    y_index=None,
    output_file="peak_amplitudes_Bz.png",
    step_stride=5
):
    """
    Tracks the amplitudes of the two strongest pulse peaks over time by
    extracting a horizontal lineout (along x) at a fixed y-index for each step.

    Saves a plot of both peak amplitudes vs time.

    Parameters:
        quantity (str): Field component to analyze (e.g., "Bz").
        folder (str): Folder with HDF5 field data and params.txt.
        y_index (int): Row index (y) to slice. If None, defaults to middle row.
        output_file (str): Output filename for the plot.
        step_stride (int): Process only every Nth step (e.g., 5 = every 5 steps)
    """
    import os
    import glob
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks

    # 1) Read simulation parameters
    params = read_params(os.path.join(folder, "params.txt"))
    box_x = params.get("box_x", 10.0)
    nx_global = int(params.get("nx_global", 180))
    ny_global = int(params.get("ny_global", 180))
    guard = int(params.get("guard", 2))
    interior_nx = int(params.get("interior_nx", 30))
    interior_ny = int(params.get("interior_ny", 30))
    dt = params.get("dt", 1.0)

    if y_index is None:
        y_index = ny_global // 2

    # 2) Get sorted list of available step numbers
    step_files = sorted(glob.glob(os.path.join(folder, "fields_rank_0_step_*.h5")))
    steps = []
    for f in step_files:
        try:
            step_part = f.split("_step_")[1].split(".h5")[0]
            step = int(step_part.strip())
            steps.append(step)
        except (IndexError, ValueError):
            print(f"⚠️ Skipping malformed file: {f}")
    steps = sorted(steps)[::step_stride]  # apply step skipping here

    times = []
    peak1 = []
    peak2 = []

    for s in steps:
        field_2d = load_field(s, folder, quantity, box_x, box_x,
                              nx_global, ny_global, guard,
                              interior_nx, interior_ny)
        lineout = field_2d[y_index, :]

        # Find local maxima
        peaks, _ = find_peaks(lineout, distance=10)
        peak_vals = lineout[peaks] if len(peaks) >= 1 else np.array([0])

        # Take the top 2 peak values (if available)
        top_peaks = sorted(peak_vals, reverse=True)[:2]
        if len(top_peaks) == 1:
            top_peaks.append(0.0)
        elif len(top_peaks) == 0:
            top_peaks = [0.0, 0.0]

        times.append(s * dt)
        peak1.append(top_peaks[0])
        peak2.append(top_peaks[1])

    # 3) Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, peak1, label="1st peak amplitude", lw=2)
    plt.plot(times, peak2, label="2nd peak amplitude", lw=2)

    plt.title(f"Peak amplitudes of ${quantity}$ vs time using nx = ny = {nx_global}", fontsize=18)
    plt.xlabel("$t \\, [\\omega_p^{-1}]$", fontsize=16)
    plt.ylabel(f"${quantity} \\, [m_e \\, c \\, \\omega_p / e]$", fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.show()

    print(f"Peak amplitude plot saved to '{output_file}'.")


##############################################################################
# 7) Main script logic
##############################################################################

if __name__ == "__main__":

    # a) Read parameters from 'params.txt' if it exists:
    params = read_params("Simulation/Fields/params.txt")

    # Provide fallback defaults if params.txt doesn't exist or lacks a key
    box_x = params.get("box_x", 10.0)
    box_y = params.get("box_y", 10.0)
    nx_global = params.get("nx_global", 180)
    ny_global = params.get("ny_global", 180)
    guard = params.get("guard", 2)
    interior_nx= params.get("interior_nx", 30)
    interior_ny= params.get("interior_ny", 30)
    dt = params.get("dt", 1.0)

    # Print simulation time step
    print(f"dt from params.txt: {dt} [1/ωₚ]")

    # b) Which step do you want to load and plot?
    step_to_load = 250                            # ----------------------------------------> Change this to plot different steps
                                                                                            #(The step must exist or else the plots show 0)

    # c) Reconstruct each field
    ex_data = load_field(step=step_to_load,
                         folder="Simulation/Fields",
                         quantity="Ex",
                         box_x=box_x,
                         box_y=box_y,
                         nx_global=nx_global,
                         ny_global=ny_global,
                         guard=guard,
                         interior_nx=interior_nx,
                         interior_ny=interior_ny)

    ey_data = load_field(step=step_to_load,
                         folder="Simulation/Fields",
                         quantity="Ey",
                         box_x=box_x,
                         box_y=box_y,
                         nx_global=nx_global,
                         ny_global=ny_global,
                         guard=guard,
                         interior_nx=interior_nx,
                         interior_ny=interior_ny)

    ez_data = load_field(step=step_to_load,
                         folder="Simulation/Fields",
                         quantity="Ez",
                         box_x=box_x,
                         box_y=box_y,
                         nx_global=nx_global,
                         ny_global=ny_global,
                         guard=guard,
                         interior_nx=interior_nx,
                         interior_ny=interior_ny)

    bx_data = load_field(step=step_to_load,
                         folder="Simulation/Fields",
                         quantity="Bx",
                         box_x=box_x,
                         box_y=box_y,
                         nx_global=nx_global,
                         ny_global=ny_global,
                         guard=guard,
                         interior_nx=interior_nx,
                         interior_ny=interior_ny)

    by_data = load_field(step=step_to_load,
                         folder="Simulation/Fields",
                         quantity="By",
                         box_x=box_x,
                         box_y=box_y,
                         nx_global=nx_global,
                         ny_global=ny_global,
                         guard=guard,
                         interior_nx=interior_nx,
                         interior_ny=interior_ny)

    bz_data = load_field(step=step_to_load,
                         folder="Simulation/Fields",
                         quantity="Bz",
                         box_x=box_x,
                         box_y=box_y,
                         nx_global=nx_global,
                         ny_global=ny_global,
                         guard=guard,
                         interior_nx=interior_nx,
                         interior_ny=interior_ny)

    # d) Plot them one by one
    #plot_field(ex_data, box_x=box_x, box_y=box_y, quantity="Ex", dt=dt, step=step_to_load)
    #plot_field(ey_data, box_x=box_x, box_y=box_y, quantity="Ey", dt=dt, step=step_to_load)
    #plot_field(ez_data, box_x=box_x, box_y=box_y, quantity="Ez", dt=dt, step=step_to_load)
    #plot_field(bx_data, box_x=box_x, box_y=box_y, quantity="Bx", dt=dt, step=step_to_load)
    #plot_field(by_data, box_x=box_x, box_y=box_y, quantity="By", dt=dt, step=step_to_load)
    #plot_field(bz_data, box_x=box_x, box_y=box_y, quantity="Bz", dt=dt, step=step_to_load)

    # e) Create a movie with the evolution of the field components
    #create_field_animation(quantity="Ex", folder="Simulation/Fields", output_folder="Simulation_Videos", fps=20)
    #create_field_animation(quantity="Ey", folder="Simulation/Fields", output_folder="Simulation_Videos", fps=20)
    #create_field_animation(quantity="Bz", folder="Simulation/Fields", output_folder="Simulation_Videos", fps=20)

    # f) Create the lineout plots and Error diagnostics
    desired_steps = [0, 5150, 15400, 30750, 62650] # nx = 450
    #desired_steps = [0, 8250, 24650, 49200, 100250] # nx = 720
    plot_line_slices_along_x_steps( steps_to_plot=desired_steps, quantity="Bz", folder="Simulation/Fields", y_index=None, output_file="Simulation_Videos/line_slices_Bz_2_nx=450.pdf")

    # g) Plot the 2 peak amplitudes of the Bz pulse over time
    track_peak_amplitudes_over_time(
        quantity="Bz",
        folder="Simulation/Fields",
        y_index=None,
        output_file="Simulation_Videos/peak_amplitudes_Bz_nx=450.pdf",
        step_stride= 49
    )