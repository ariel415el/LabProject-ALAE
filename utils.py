import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.cm as cm
import os


class simple_logger(object):
    def __init__(self, fname):
        self.file = open(fname, "w")

    def log(self, txt, end="\n", stdout=False):
        self.file.write(txt + end)
        if stdout:
            print(txt, end=end)


def plot_training(losses, names, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig = plt.figure(figsize=(10, 6))
    for i, (loss_list, name) in enumerate(zip(losses, names)):
        ax = fig.add_subplot(1, len(losses), 1 + i)
        ax.set_title(name)
        ax.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(plot_path)
    plt.clf()


def plot_tsne(data, labels, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    clustered_data = TSNE(n_components=2, perplexity=40, n_iter=300).fit_transform(data)
    possible_labels = np.sort(np.unique(labels))
    colors = cm.rainbow(np.linspace(0, 1, len(possible_labels)))
    for label in possible_labels:
        label_data = clustered_data[labels == label]
        plt.scatter(label_data[:, 0], label_data[:, 1], color=colors[label], label=label)
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()


def plot_training_accuracies(train_accuracies, test_accuracies, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    plt.plot(np.arange(len(train_accuracies)), train_accuracies, label='train-acc', c='r')
    plt.plot(np.arange(len(test_accuracies)), test_accuracies, label='test-acc', c='b')
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()


def visualize_classification(data, predictions, labels, plot_path):
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    num_samples_root = int(np.sqrt(len(data)))
    num_samples = int(num_samples_root**2)
    img_size = int(np.sqrt(len(data[0])))
    fig = plt.figure(figsize=(15, 10))
    for i in range(num_samples):
        ax = fig.add_subplot(num_samples_root, num_samples_root, 1 + i)
        ax.imshow(data[i].reshape(img_size, img_size))
        ax.set_title(f"GT: {labels[i]}; Prediction: {predictions[i]}")
        ax.set_xticks([])
    plt.savefig(plot_path)
    plt.clf()


def plot_examples(autoencoders, data, N=10, plot_path=None):

    im_dim = int(np.sqrt(len(data[0])))
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    indices = np.random.randint(len(data), size=N)
    fig = plt.figure()
    for i in range(N):
        sample = data[indices[i]]
        ax = fig.add_subplot(2 + len(autoencoders), N, 1+i)
        ax.imshow(sample.reshape(im_dim, im_dim), cmap='gray')
        if i == N // 2:
            ax.set_title(f"original", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for j, ae in enumerate(autoencoders):
        for i in range(N):
            sample = data[indices[i]]
            ax = fig.add_subplot(2 + len(autoencoders), N, 1 + (j + 1) * N + i)
            reconstructed = ae.decode(ae.encode(sample)).reshape(im_dim, im_dim)
            reconstructed = np.clip(reconstructed, 0, 1)
            if i == N // 2:
                ax.set_title(f"{ae}", fontsize=10)
            ax.imshow(reconstructed, cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(plot_path)
    plt.clf()


def plot_latent_interpolation(ae, data, N=5, plot_path=None):
    im_dim = int(np.sqrt(len(data[0])))
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    indices = np.random.randint(len(data), size=2)
    start = data[indices[0]]
    end = data[indices[1]]
    start_z = ae.encode(start)
    end_z = ae.encode(end)

    fig = plt.figure()
    for i, c in enumerate(np.arange(0, 1 + 1 / N, 1 / (N-1))):
        interpollated_img = (1-c) * start + c * end
        ax = fig.add_subplot(3, N,  1 + i)
        ax.set_title(f"Data {c}")
        ax.imshow(interpollated_img.reshape(im_dim, im_dim), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

        interpollated_z = (1-c) * start_z + c * end_z
        img = np.clip(ae.decode(interpollated_z), 0 , 1)
        ax = fig.add_subplot(2, N, N + 1 + i)
        ax.set_title(f"Latent {c}")
        ax.imshow(img.reshape(im_dim, im_dim), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

    plt.savefig(plot_path)
    plt.clf()
