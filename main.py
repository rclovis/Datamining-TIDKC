import TrajClustering as tc
from IDK import *

if __name__ == "__main__":
    traj = tc.TrajClustering()
    traj.load_dataset("geolife")
    traj.run_distance("IDK2")
    traj.plot_mds()
    traj.run_clustering("Spectral", 10)
    traj.plot_clusters()
