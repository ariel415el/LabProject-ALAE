import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class EncodingsDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, index):
        # stuff
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.data.shape[0]

    def get_input_shape(self):
        return self.data.shape[1]


# define NN architecture
class MLP(nn.Module):
    def __init__(self, input_shape=8 * 8, hidden_dim=512):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.droput(x)
        x = F.relu(self.fc2(x))
        x = self.droput(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x


def train(model, train_loader, optimizer, epoch, log_interval=10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        # import pdb;pdb.set_trace()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if log_interval and batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, lr{:.5f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)
    # print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy))
    return accuracy


def train_on_data(train_data_and_labels, test_data_and_labels,
                  batch_size=64, epochs=5, lr=0.01, lr_decay=0.9, log_interval=None, plot_path=None):
    kwargs = {'batch_size': batch_size}
    if device != "cpu":
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    train_dataset = EncodingsDataset(*train_data_and_labels)
    test_dataset = EncodingsDataset(*test_data_and_labels)
    train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

    model = MLP(input_shape=train_dataset.get_input_shape()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)

    train_accuracies, test_accuracies = [], []
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, log_interval)
        scheduler.step()
        train_accuracies += [test(model, train_loader)]
        test_accuracies += [test(model, test_loader)]
        if log_interval:
            print(f"Epoch {epoch}, train/test accuracy {train_accuracies[-1]}/{test_accuracies[-1]}")

    # torch.save(model.state_dict(), "mnist_cnn.pt")

    if plot_path is not None:
        plt.plot(np.arange(len(train_accuracies)), train_accuracies,label='train-acc', c='r')
        plt.plot(np.arange(len(test_accuracies)), test_accuracies, label='test-acc', c='b')
        plt.legend()
        plt.savefig(plot_path)
        plt.clf()

    return train_accuracies[-1], test_accuracies[-1]



if __name__ == '__main__':
    import sys
    import os
    import inspect

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currentdir)
    sys.path.insert(0, parentdir)
    from Linear_encoding.datasets import get_mnist
    train_data, train_labels, test_data, test_labels, dataset_name = get_mnist(data_dir='Linear_encoding/data')

    train_accuracies, test_accuracies = train_on_data((train_data, train_labels), (test_data, test_labels), epochs=5, lr=0.001, lr_decay=0.7)
    plt.plot(np.arange(len(train_accuracies)), train_accuracies,label='train-acc', c='r')
    plt.plot(np.arange(len(test_accuracies)), test_accuracies, label='test-acc', c='b')
    plt.show()