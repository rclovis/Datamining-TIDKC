import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


def visualize_trajectory(data, labels):
    """ 
    Visualize the trajectory data with labels 

    Args:
        data: list of numpy arrays, each array represents a trajectory

        labels: list of integers, each integer represents the label of a trajectory. 
            This can be user-defined, as the results of clustering algorithms
    """

    # Create a colormap to map labels to colors
    cmap = plt.get_cmap('viridis', np.max(labels) - np.min(labels) + 1)

    # Create a normalization for the labels
    norm = mcolors.Normalize(vmin=np.min(labels), vmax=np.max(labels))

    # Plot each trajectory
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, trajectory in enumerate(data):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=cmap(norm(labels[i])))

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'Trajectory Data')
    ax.grid(True)

    # Create a mappable object for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Add the colorbar to the plot
    fig.colorbar(sm, ax=ax, ticks=range(np.min(labels),np.max(labels)+1), label='Labels')

    plt.show()