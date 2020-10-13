import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
import numpy as np
from sklearn import svm
from utils import plot_training_accuracies

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Evaluateor:
    def __init__(self):
        self.name = "Abstract-Evaluateor"

    def evaluate(self, ae, data, projected_data, plot_path):
        raise NotImplementedError

    def __str__(self):
        return self.name


class ReconstructionLossEvaluator(Evaluateor):
    def __init__(self):
        super(ReconstructionLossEvaluator, self).__init__()
        self.name = "ReconstructionLoss-(L2)"

    def evaluate(self, ae, data, projected_data, plot_path):
        train_data, _, test_data, _ = data
        return f"{ae.get_reconstuction_loss(train_data):.4f}/{ae.get_reconstuction_loss(test_data):.4f}"


class OrthonormalityEvaluator(Evaluateor):
    def __init__(self):
        super(OrthonormalityEvaluator, self).__init__()
        self.name = "OrthonormalityEvaluator"

    def evaluate(self, ae, data, projected_data, plot_path):
        return f"{ae.get_orthonormality_loss():.4f}"


class SVMClassification(Evaluateor):
    def __init__(self, SVC_C):
        super(SVMClassification, self).__init__()
        self.name = "SVC-classfication"
        self.classifier = svm.SVC(C=SVC_C, kernel="linear")

    def evaluate(self, ae, data, projected_data, plot_path):
        train_data, train_labels, test_data, test_labels = data
        projected_train_data, projected_test_data = projected_data
        self.classifier.fit(projected_train_data, train_labels)
        train_accuracy = np.mean(train_labels == self.classifier.predict(projected_train_data))*100
        test_accuracy = np.mean(test_labels == self.classifier.predict(projected_test_data)) * 100
        return f"{train_accuracy:.2f}/{test_accuracy:.2f}"


def compute_1nn_accuracy(data, labels):
    predictions = np.ones(data.shape[0]) * -1
    for j in range(data.shape[0]):
        # distances = np.sqrt(np.sum((data - data[j])**2, axis=1)) # l2 distance
        distances = np.sqrt(np.sum((data - data[j])**2, axis=1)) # l2 distance
        distances[j] = np.inf
        predictions[j] = labels[np.argmin(distances)]
    return 100 * np.mean(labels == predictions)


class FirstNearestNeighbor(Evaluateor):
    def __init__(self):
        super(FirstNearestNeighbor, self).__init__()
        self.name = "1NN-classification"

    def evaluate(self, ae, data, projected_data, plot_path):
        train_data, train_labels, test_data, test_labels = data
        projected_train_data, projected_test_data = projected_data
        return f"{compute_1nn_accuracy(projected_train_data, train_labels):.2f}/{compute_1nn_accuracy(projected_test_data, test_labels):.2f}"


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


def train_epoch(model, train_loader, optimizer, epoch, log_interval=10):
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


class MLP_classification(Evaluateor):
    def __init__(self, batch_size=64, epochs=5, lr=0.01, lr_decay=0.9, log_interval=None):
        super(MLP_classification, self).__init__()
        self.name = "MLP-classification"
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.lr_decay = lr_decay
        self.log_interval = log_interval

    def evaluate(self, ae, data, projected_data, plot_path):
        dataset = (projected_data[0], data[1], projected_data[1], data[3])
        return self.train_mlp_classifier(dataset, plot_path)

    def train_mlp_classifier(self, data, plot_path):
        train_data, train_labels, test_data, test_labels = data

        kwargs = {'batch_size': self.batch_size}
        if device != "cpu":
            kwargs.update({'num_workers': 1,
                           'pin_memory': True,
                           'shuffle': True},
                          )

        train_dataset = EncodingsDataset(train_data, train_labels)
        test_dataset = EncodingsDataset(test_data, test_labels)
        train_loader = torch.utils.data.DataLoader(train_dataset, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_dataset, **kwargs)

        model = MLP(input_shape=train_dataset.get_input_shape()).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.lr_decay)

        train_accuracies, test_accuracies = [], []
        for epoch in range(1, self.epochs + 1):
            train_epoch(model, train_loader, optimizer, epoch, self.log_interval)
            scheduler.step()
            train_accuracies += [test(model, train_loader)]
            test_accuracies += [test(model, test_loader)]
            if self.log_interval:
                print(f"Epoch {epoch}, train/test accuracy {train_accuracies[-1]}/{test_accuracies[-1]}")

        # torch.save(model.state_dict(), "mnist_cnn.pt")

        if plot_path is not None:
            plot_training_accuracies(train_accuracies, test_accuracies, plot_path)

        return f"{train_accuracies[-1]:.2f}/{test_accuracies[-1]:.2f}"

