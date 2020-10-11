import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_scatter():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    N_plt = 1000

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_x[:N_plt, 0], train_x[:N_plt, 1], train_y[:N_plt])

    plt.show()

def plot_histogram():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    plt.hist(train_y,bins=40)
    plt.show()


if __name__ == "__main__":
    plot_scatter()
    plot_histogram()
