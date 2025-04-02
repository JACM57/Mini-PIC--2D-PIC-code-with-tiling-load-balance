import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Draw the main grid cell
ax.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], 'k')  # outer square
ax.set_aspect('equal')

# Draw dashed lines (cross in center)
ax.plot([0.5, 0.5], [0, 1], 'k--')
ax.plot([0, 1], [0.5, 0.5], 'k--')

# Field positions and labels with colors
fields = {
    (0, 0): (r'$E_z$', 'blue'),
    (0.5, 0): (r'$E_x,\ B_y$', 'green'),
    (0, 0.5): (r'$E_y,\ B_x$', 'red'),
    (0.5, 0.5): (r'', 'orange')  # center circle only, label moved right
}

# Draw circles
for (x, y), (_, color) in fields.items():
    circle = patches.Circle((x, y), 0.03, edgecolor=color, facecolor='none', linewidth=1.8)
    ax.add_patch(circle)

# Add labels
ax.text(0, -0.08, r'$E_z$', fontsize=14, ha='center', va='center', color='blue')
ax.text(0.5, -0.08, r'$E_x,\ B_y$', fontsize=14, ha='center', va='center', color='green')
ax.text(-0.08, 0.5, r'$E_y,\ B_x$', fontsize=14, ha='center', va='center', color='red', rotation=90)
ax.text(0.56, 0.46, r'$B_z$', fontsize=14, ha='left', va='center', color='orange')
ax.text(0.01, 0.05, r'$(i,j)$', fontsize=14, ha='left', va='center', color='black')


# Arrows for dx and dy
ax.annotate('', xy=(0, 1.05), xytext=(1, 1.05), arrowprops=dict(arrowstyle='<->'))
ax.text(0.5, 1.08, r'$dx$', fontsize=14, ha='center')

ax.annotate('', xy=(1.05, 0), xytext=(1.05, 1), arrowprops=dict(arrowstyle='<->'))
ax.text(1.08, 0.5, r'$dy$', fontsize=14, va='center', rotation=90)

# Remove axes
ax.axis('off')

# Show plot
plt.tight_layout()
plt.savefig("Report_Plots/Field_Staggering.pdf")
plt.show()
