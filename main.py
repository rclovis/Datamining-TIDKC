import TrajClustering as tc
from IDK import *

if __name__ == "__main__":
    traj = tc.TrajClustering()
    traj.load_dataset("TRAFFIC")
    # traj.run_distance("IDK2")
    # traj.plot_mds()
    traj.run_clustering("TIDKC", 11)
    traj.plot_clusters()
