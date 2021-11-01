import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    root = './result/sketch/loss_txt'
    n = len(os.listdir(root))

    Semi = []
    epoch = []
    for i in range(n):
        fileName = os.path.join(root, 'epoch_' + str(i) + '.txt')
        m = []
        with open(fileName, 'r') as f:
            for line in f.readlines():
                Semi.append(float(line.split()[4].split(':')[-1][:-1]))
                m.append(float(line.split()[4].split(':')[-1][:-1]))
        epoch.append(np.round(np.mean(np.array(m)), 4))

    # Semi = Semi[:]
    # plt.plot(range(len(Semi)), Semi)
    # plt.xlabel('iterations')
    # plt.ylabel('SemiLoss')
    # plt.title('1000 - 7000 iteration')
    # plt.show()

    # ax = plt.axes(xlim=[0, 30])
    # ax.grid()
    plt.plot(range(len(epoch)), epoch)
    plt.xlabel('epoch')
    plt.ylabel('SemiLoss Value')
    plt.show()