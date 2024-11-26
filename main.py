from TrajClustering import TrajClustering


def main():
    tc = TrajClustering()
    tc.load_dataset("TRAFFIC")
    tc.run_distance("IDK")
    tc.plot_mds()
    tc.run_clustering("Spectral", 10)
    tc.plot_clusters()


if __name__ == "__main__":
    main()
