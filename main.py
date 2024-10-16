from IDK.new_inne import *
import scipy.io as io
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == "__main__":
    raw_mat = io.loadmat("./datasets/geolife.mat")
    data = np.array(raw_mat['data'][0])
    data = data[:100]
    label = raw_mat['label'][0]
    # for i in range(len(data)):
    #     plt.scatter(data[i][:, 0], data[i][:, 1])
    #     plt.plot(data[i][:, 0], data[i][:, 1])
    # plt.show()

    alldata = []
    for i in range(len(data)):
        for data_point in data[i]:
            alldata.append(data_point)
    alldata = np.array(alldata)

    colors = np.array(["#bababa", "#43cc5c"])

    inne = IsolationNNE(random_seed=42)
    result = inne.train(alldata).predict(alldata)
    print(result)
    plt.scatter(alldata[:, 0], alldata[:, 1], s=10, color=colors[result])

    # inne = iNN_IK(16, 200)
    # all_ikmap = inne.fit_transform(alldata).toarray()

    # y_pred = np.where(np.mean(all_ikmap, axis=1) < 0.04, -1, 1)
    # plt.scatter(alldata[:, 0], alldata[:, 1], s=10, color=colors[(y_pred + 1) // 2])

    plt.show()
