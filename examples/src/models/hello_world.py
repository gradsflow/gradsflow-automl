#  Copyright (c) 2021 GradsFlow. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# Source code inspired from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from gradsflow import AutoDataset, Model
from gradsflow.callbacks import CSVLogger, ModelCheckpoint

# Replace dataloaders with your custom dataset, and you are all set to train your model
image_size = (64, 64)
batch_size = 4

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root="~/data", train=True, download=True, transform=transform)
train_dl = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root="~/data", train=False, download=True, transform=transform)
val_dl = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
num_classes = len(trainset.classes)
cbs = [
    CSVLogger(
        verbose=True,
    ),
    ModelCheckpoint(),
    # EmissionTrackerCallback(),
    # CometCallback(offline=True),
    # WandbCallback(),
]


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    autodataset = AutoDataset(train_dl, val_dl, num_classes=num_classes)
    net = Net()
    model = Model(net)
    criterion = nn.CrossEntropyLoss()

    model.compile(
        criterion,
        optim.SGD,
        optimizer_config={"momentum": 0.9},
        learning_rate=0.001,
        metrics=[MulticlassAccuracy(autodataset.num_classes)],
    )
    model.fit(autodataset, max_epochs=2, callbacks=cbs)

    dataiter = iter(val_dl)
    images, labels = next(dataiter)

    # print images
    # imshow(torchvision.utils.make_grid(images))
    print("GroundTruth: ", " ".join(f"{trainset.classes[labels[j]]:5s}" for j in range(4)))

    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    print("Predicted: ", " ".join(f"{trainset.classes[predicted[j]]:5s}" for j in range(4)))
