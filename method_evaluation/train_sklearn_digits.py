import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

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
    def __init__(self, input_shape=8 * 8):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.fc1 = nn.Linear(input_shape, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.droput = nn.Dropout(0.2)

    def forward(self, x):
        # x = x.view(-1, self.input_shape)
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
        # if batch_idx % log_interval == 0:
        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, lr{:.5f}'.format(
        #         epoch, batch_idx * len(data), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item(), optimizer.param_groups[0]['lr']))


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


def train_sklearn_digits(data, labels, batch_size=64, epochs=5, lr=0.01, lr_decay=0.9, log_interval=10):
    kwargs = {'batch_size': batch_size}
    if device != "cpu":
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    dataset = EncodingsDataset(data, labels)
    train_loader = torch.utils.data.DataLoader(dataset, **kwargs)

    model = MLP(input_shape=dataset.get_input_shape()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
    for epoch in range(1, epochs + 1):
        train(model, train_loader, optimizer, epoch, log_interval)
        scheduler.step()
        accuracy = test(model, train_loader)

    # torch.save(model.state_dict(), "mnist_cnn.pt")

    return accuracy

if __name__ == '__main__':
    train_sklearn_digits(epochs=15)
