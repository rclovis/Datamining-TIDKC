import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as io

# Assuming 'data' is a list of arrays, where each array is a trajectory
# Each trajectory should be an array of shape (n_points, 2) where columns are latitude and longitude

# Load your data
raw_mat = io.loadmat("./datasets/geolife.mat")
data = raw_mat['data'][0]  # Assuming data is structured this way

# Plot each trajectory
plt.figure(figsize=(10, 10))
for trajectory in data:
    plt.plot(trajectory[:, 0], trajectory[:, 1], label='data')  # longitude is x, latitude is y

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Trajectory Data')
plt.grid(True)
plt.show()
