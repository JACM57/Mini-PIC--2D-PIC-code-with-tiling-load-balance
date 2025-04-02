import numpy as np
import matplotlib.pyplot as plt

# Load the field data from files
t0 = np.loadtxt("field_t0.txt")
t1 = np.loadtxt("field_t1.txt")
t2 = np.loadtxt("field_t2.txt")
t3 = np.loadtxt("field_t3.txt")

# Position array (assuming uniform spacing)
Ny = len(t0)  # Get grid size from file
y = np.linspace(0, Ny - 1, Ny)  # Index positions

# Plot all four graphs
plt.figure(figsize=(10, 6))

plt.subplot(2, 2, 1)
plt.plot(y, t0, label="t = 0")
plt.xlabel("Position (grid index)")
plt.ylabel("Ez (V/m)")
plt.title("Initial Field")
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(y, t1, label="t = T/4", color="orange")
plt.xlabel("Position (grid index)")
plt.ylabel("Ez (V/m)")
plt.title("Quarterway")
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(y, t2, label="t = T/2", color="green")
plt.xlabel("Position (grid index)")
plt.ylabel("Ez (V/m)")
plt.title("Halfway")
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(y, t3, label="t = T", color="red")
plt.xlabel("Position (grid index)")
plt.ylabel("Ez (V/m)")
plt.title("Final Field")
plt.grid()

plt.tight_layout()
plt.show()
