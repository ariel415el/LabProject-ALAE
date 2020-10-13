import os
import matplotlib.pyplot as plt
import numpy as np
COLORS =['r', 'g', 'b']


def plot_training(losses, names, plots_dir, plot_name):
    fig = plt.figure(figsize=(10, 6))
    for i, (loss_list, name) in enumerate(zip(losses, names)):
        ax = fig.add_subplot(1, len(losses), 1 + i)
        ax.set_title(name)
        ax.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(os.path.join(plots_dir, plot_name + ".png"))
    plt.clf()


class simple_logger(object):
    def __init__(self, fname):
        self.file = open(fname, "w")

    def log(self, txt, end="\n", stdout=True):
        self.file.write(txt + end)
        if stdout:
            print(txt, end=end)
