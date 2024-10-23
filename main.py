from IDK import *
import scipy.io as io
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == "__main__":
    raw_mat = io.loadmat("./datasets/CASIA.mat")
    data = np.array(raw_mat['data'][0])
    data = data[:100]

    idk = IDK(random_seed=42)
    result = idk.idk(data)

    colors = np.array(["#000000", "#43cc5c"])
    for i in range(len(data)):
        plt.plot(data[i][:, 0], data[i][:, 1], color=plt.cm.jet(result[i]))
    plt.show()