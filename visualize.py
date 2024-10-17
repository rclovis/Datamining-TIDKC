import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import scipy.io as io
import numpy as np
import os
import pandas as pd


def load_and_preprocess_data(dataset_name):
    """ 
    name of 7 datasets: CASIA, cross, geolife, pedes3, pedes4, TRAFFIC, cyclists

    cyclists dataset is not in .mat format, although it has been converted to .mat using scipy.io.savemat, 
        it is still constructed every time it is loaded from the .csv files

    TRAFFIC dataset has a different structure, so we need to handle it separately
    """

    # construct data and labels for cyclists dataset
    if dataset_name == 'cyclists':
        base_dir = './datasets/cyclists'
        classes = {'moving': 0, 'starting': 1, 'stopping': 2, 'waiting': 3}
        data = []
        labels = []

        for cls, label in classes.items():
            class_dir = os.path.join(base_dir, cls)
            for file in os.listdir(class_dir):
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(class_dir, file))
                    trajectory = df[['x', 'y']].values
                    data.append(trajectory)
                    labels.append(label)

        return data, labels

    # Load the dataset
    raw_mat = io.loadmat(f'./datasets/{dataset_name}.mat')

    data = raw_mat['data'] if dataset_name == 'TRAFFIC' else  raw_mat['data'][0]

    # since all datasets have labels, we can use the key 'class' or 'label' to access the labels
    label_key = 'class' if 'class' in raw_mat.keys() else 'label'
    labels = raw_mat[label_key][0]

    return data, labels

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

data, labels = load_and_preprocess_data('cyclists')
visualize_trajectory(data, labels)