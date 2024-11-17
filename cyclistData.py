import numpy as np
import pandas as pd
import scipy.io


# Function to load and process data from the .mat file
def load_mat_data(file_path):
    data = scipy.io.loadmat(file_path)

    # Extracting the 'data' field
    trajectories = data["data"]

    # Calculate number of trajectories, min, max, and average trajectory lengths
    num_trajectories = trajectories.shape[0]
    trajectory_lengths = [len(traj) for traj in trajectories]

    min_length = min(trajectory_lengths)
    max_length = max(trajectory_lengths)
    avg_length = np.mean(trajectory_lengths)

    total_points = sum(trajectory_lengths)

    return total_points, num_trajectories, min_length, max_length, avg_length


# Dataset paths and display names
datasets = {
    "VRU_pedes_3": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\pedes3.mat",
    "VRU_pedes_4": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\pedes4.mat",
    "VRU_cyclists": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\cyclists1.mat",
    "TRAFFIC": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\TRAFFIC.mat",
    "CROSS": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\cross.mat",
    "CASIA": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\CASIA.mat",
    "Geolife": r"C:\Users\hp\Datamining\Datamining-TIDKC\datasets\geolife.mat",
}

# Summary table for results
summary_table = {
    "Dataset Name": [],
    "Total Points": [],
    "Number of Trajectories": [],
    "Min Trajectory Length": [],
    "Max Trajectory Length": [],
    "Avg Trajectory Length": [],
}

# Process each dataset
for display_name, file_path in datasets.items():
    total_points, num_trajectories, min_length, max_length, avg_length = load_mat_data(
        file_path
    )

    summary_table["Dataset Name"].append(display_name)
    summary_table["Total Points"].append(total_points)
    summary_table["Number of Trajectories"].append(num_trajectories)
    summary_table["Min Trajectory Length"].append(min_length)
    summary_table["Max Trajectory Length"].append(max_length)
    summary_table["Avg Trajectory Length"].append(avg_length)

# Convert the summary to a DataFrame for easy viewing
summary_df = pd.DataFrame(summary_table)

# Print the summary table
print(summary_df)

# Export the summary table to CSV (if needed)
summary_df.to_csv("dataset_summary.csv", index=False)
