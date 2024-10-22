import scipy.io as io
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