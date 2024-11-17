import matplotlib
import matplotlib.pyplot as plt
import scipy.io as io

import TrajClustering as tc
from IDK import *

# plt.style.use('ggplot')

# if __name__ == "__main__":
#     raw_mat = io.loadmat("./datasets/cross.mat")
#     data = np.array(raw_mat['data'][0])
#     # data = data[:100]
#     idk = IDK(random_seed=42)
#     result = idk.idk_square(data)
#     colors = np.array(["#000000", "#43cc5c"])
#     for i in range(len(data)):
#         plt.plot(data[i][:, 0], data[i][:, 1], color=plt.cm.jet(result[i]))
#     plt.show()

if __name__ == "__main__":
    traj = tc.TrajClustering()
    traj.load_dataset("TRAFFIC")
    # traj.run_distance("IDK2")
    # traj.plot_mds()
    traj.run_clustering("TIDKC", 11)
    traj.plot_clusters()
