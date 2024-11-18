from TrajClustering import TrajClustering

if __name__ == "__main__":
    traj = TrajClustering()
    traj.load_dataset("TRAFFIC")
    traj.run_clustering("TIDKC", number_of_clusters=11)
    traj.plot_clusters()
