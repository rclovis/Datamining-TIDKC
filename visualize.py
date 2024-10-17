import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as io
import numpy as np

# Load your data
raw_mat = io.loadmat("./datasets/geolife.mat")
data = raw_mat['data'][0]  # Assuming data is structured this way
labels = raw_mat['label'][0]  # Assuming labels are structured this way

# Create a colormap to map labels to colors
cmap = plt.get_cmap('viridis', np.max(labels) - np.min(labels) + 1)

# Create a normalization for the labels
norm = mcolors.Normalize(vmin=np.min(labels), vmax=np.max(labels))

# Plot each trajectory
fig, ax = plt.subplots(figsize=(10, 10))
for i, trajectory in enumerate(data):
    ax.plot(trajectory[:, 0], trajectory[:, 1], color=cmap(norm(labels[i])))

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Trajectory Data')
ax.grid(True)

# Create a mappable object for the colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Add the colorbar to the plot
fig.colorbar(sm, ax=ax, ticks=range(np.min(labels),np.max(labels)+1), label='Labels')

plt.show()
