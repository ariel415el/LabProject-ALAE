from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt


def get_swiss_roll(plot=False):
    '''
    Example code to show you how to load the swiss roll data and plot it.
    '''

    # load the dataset:
    X, color = datasets.make_swiss_roll(n_samples=4000)

    if plot:
        # plot the data:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral,s=0.6)
        plt.show()
    return X, color


def get_digits(plot=False):
    '''
    Example code to show you how to load the MNIST data and plot it.
    '''

    # load the 8x8 digits  data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target
    if plot:
        # plot examples:
        plt.gray()
        for i in range(10):
            plt.subplot(2, 5, i+1)
            plt.axis('off')
            plt.imshow(np.reshape(data[i, :], (8, 8)))
            plt.title("Digit " + str(labels[i]))
        plt.show()
    return data, labels


def get_synthetic_embedded_data():
    N=5000 # numer of samples
    xs = np.linspace(-7, 7, N)
    ys = 5 * np.sin(0.5*xs)

    data_2d = np.stack([xs,ys],axis=1)

    # Create a random projectrion from 2d to 3d
    q, _ = np.linalg.qr(np.random.normal(size=(3,3)))

    data_3d = np.dot(data_2d, q[:2])

    data_3d += np.random.normal(-0.7, 0.7, size=data_3d.shape)
    colors = np.arange(N)
    return data_3d, colors, data_2d


def get_multi_gaussian_2d_data(plot=False):
    N = 1000
    data = np.random.multivariate_normal([5,-2], [[1,-1.5],[-1.5,2]], N)
    colors = np.arange(N)
    if plot:
        plt.scatter(data[:, 0], data[:, 1], c=colors, cmap=plt.cm.Spectral, s=0.1)
        plt.xlim(0, 50)
        plt.ylim(0, 50)
        plt.show()
    return data, colors


def get_multi_gaussian_3d_data(plot=False):
    N = 1000
    data = np.random.multivariate_normal([5,-2, 1], [[0.05,0,0],[0,0.5,0],[0,0,4]], N)
    colors = np.arange(N)
    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(data[:, 0], data[:, 1], c=colors, cmap=plt.cm.Spectral, s=0.1, marker='o')
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 50)
        plt.show()
    return data, colors