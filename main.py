from IDK.new_inne import *
import scipy.io as io
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == "__main__":
    raw_mat = io.loadmat("./datasets/CASIA.mat")
    data = np.array(raw_mat['data'][0])
    data = data[:100]

    alldata = []
    index_lines = np.array([0])
    for i in range(len(data)):
        for data_point in data[i]:
            alldata.append(data_point)
        index_lines = np.append(index_lines, len(alldata))
    alldata = np.array(alldata)

    colors = np.array(["#000000", "#43cc5c"])

    inne = IsolationNNE(random_seed=42)
    result = inne.generate_centroid(alldata).predict(alldata)

    idkmap = []
    for i in range(len(data)):
        idkmap.append(np.sum(result[index_lines[i]:index_lines[i + 1]], axis=0) / (index_lines[i + 1] - index_lines[i]))
    idkmap = np.array(idkmap)
    print(idkmap)
    # plt.scatter(alldata[:, 0], alldata[:, 1], s=10, color=colors[result])
    for i in range(len(data)):
        # plt.scatter(data[i][:, 0], data[i][:, 1], s=10, color=data[i])
        plt.plot(data[i][:, 0], data[i][:, 1], color=plt.cm.jet(idkmap[i]))

    plt.show()
