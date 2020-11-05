import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


def load_datapoints(path):
    datapoint = torch.from_numpy(np.load(path))
    datapoint = datapoint.view(1, 14, 32, 32, 32)
    return datapoint


dataset = torchvision.datasets.DatasetFolder(
    root='data/deepdrug3d_voxel_data_subset',
    loader=load_datapoints,
    extensions='.npy'
)

train_set, remaining = torch.utils.data.random_split(dataset, [110, 35])
validation_set, test_set = torch.utils.data.random_split(remaining, [20, 15])

train_loader = torch.utils.data.DataLoader(
    dataset=train_set,
    batch_size=4,
    shuffle=False,
    num_workers=5,
    drop_last=True
)

validation_loader = torch.utils.data.DataLoader(
    dataset=validation_set,
    batch_size=2,
    shuffle=False,
    num_workers=3,
    drop_last=True
)

test_loader = torch.utils.data.DataLoader(
    dataset=test_set,
    batch_size=1,
    shuffle=False,
    num_workers=2,
    drop_last=True
)


class ConvolutionalAE(nn.Module):
    """
        The Autoencoder network with architecture defined below
    """

    def __init__(self, **kwargs):
        super().__init__()

        # encoder
        self.conv1 = nn.Conv3d(in_channels=14, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2, 2)

        # decoder
        self.t_conv1 = nn.ConvTranspose3d(in_channels=4, out_channels=8, kernel_size=2, stride=2)
        self.t_conv2 = nn.ConvTranspose3d(in_channels=8, out_channels=14, kernel_size=2, stride=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.t_conv1(x))
        x = F.sigmoid(self.t_conv2(x))
        return x


net = ConvolutionalAE()
print(net)

# Define a Loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)


def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device


device = get_device()
print(device)
net.to(device)

# Train the network

train_loss_values = []
validation_loss_values = []

for epoch in range(2):
    train_loss = 0.0
    validation_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs = data[0].float()
        inputs = inputs.view(4, 14, 32, 32, 32)
        optimizer.zero_grad()
        outputs = net(inputs)
        print(outputs.shape)
        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    for i, data in enumerate(validation_loader, 0):
        inputs = data[0].float()
        inputs = inputs.view(2, 14, 32, 32, 32)
        outputs = net(inputs)
        loss = criterion(outputs, inputs)
        loss.backward()
        validation_loss += loss.item()

    train_loss = train_loss/len(train_loader)
    validation_loss = validation_loss/len(validation_loader)
    train_loss_values.append(train_loss)
    validation_loss_values.append(validation_loss)
    print('Epoch %d' % (epoch + 1))
    print('Train loss: %.3f' % train_loss)
    print('validation loss: %.3f\n\n' % validation_loss)

plt.plot(np.array(train_loss_values), label = "Train loss")
plt.plot(np.array(validation_loss_values), label = "Validation loss")
plt.legend()
plt.show()

print('Finished Training')


# Test network

test_loss = 0.0

for data in test_loader:
    data = data[0].float()
    data = data.view(1, 14, 32, 32, 32)
    optimizer.zero_grad()
    output = net(data)
    print(output.shape)
    loss = criterion(output, data)
    loss.backward()
    test_loss += loss.item()

test_loss = (test_loss/len(test_loader))*100
print('Test loss: %.3f %%' % (test_loss))
