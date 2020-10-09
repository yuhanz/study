import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pandas.core.common import flatten
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1, padding=1):
    # tensorflow: padding = 'same'
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv3x3transpose(in_channels, out_channels, stride=1, padding=1):
    # tensorflow: padding = 'same'
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=False)


class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,  # extra comma
    def forward(self, x):
        return x.view(*flatten([len(x), self.shape]))

class Narrow(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n,
    def forward(self, x):
        (num_records, channels, w, h) = x.shape
        return x.narrow(2,0,w-1).narrow(3,0,h-1)


# discriminator model
discriminator_model = nn.Sequential(\
    conv3x3(3, 64, (2,2)), \
    nn.BatchNorm2d(64), \
    nn.LeakyReLU(0.2), \
    conv3x3(64, 64, (2,2)), \
    nn.BatchNorm2d(64), \
    nn.LeakyReLU(0.2), \
    nn.Flatten(), \
    nn.Linear(64 * 8 * 8, 1), \
    nn.Sigmoid())
# test input:
#input = torch.from_numpy(np.random.rand(1,3,32,32).astype(np.float32))

# generator model
generator_model = nn.Sequential(\
    nn.Linear(100, 64*9*9), \
    nn.BatchNorm1d(64*9*9), \
    nn.LeakyReLU(0.2), \
    View((64,9,9)), \
    conv3x3transpose(64,64, (2,2), 1), \
    nn.BatchNorm2d(64), \
    nn.LeakyReLU(0.2), \
    conv3x3transpose(64,64, (2,2), 0), \
    nn.BatchNorm2d(64), \
    nn.LeakyReLU(0.2), \
    Narrow(1), \
    conv3x3(64, 3, 1, 0), \
    nn.Tanh())
    # result shape: 33x33
# test input:
#input = torch.from_numpy(np.random.rand(20,100).astype(np.float32))

# generator loss
gan_model = nn.Sequential(generator_model, discriminator_model)
# TODO: make descriminator not trainable


# discriminator loss
learning_rate = 0.0002
optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()


# data loader
transform = transforms.ToTensor()
batch_size = 100

train_dataset = torchvision.datasets.CIFAR10(root='./data/', \
                                             train=True, \
                                             transform=transform, \
                                             download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, \
                                           batch_size=batch_size, \
                                           shuffle=True)

# training discriminator
def trainDiscriminator():
    num_epochs = 1
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            # labels = labels.to(device)
            labels = torch.from_numpy(np.ones(batch_size).astype(np.float32))
            outputs = discriminator_model(images)
            loss = criterion(outputs, labels)
            print(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


# training generator
num_epochs = 3
for epoch in range(num_epochs):
    # TODO:
    # - select real samples
    # - generate fake samples
    # - training descripminator
    # d_loss = discriminator_model.train
    for param in discriminator_model.parameters():
        param.requires_grad = True

    trainDiscriminator()

    # TODO:
    # - generae latent points
    # - generate
    # update generator via the discriminator's error
    # g_loss = gan_model.train
    for param in discriminator_model.parameters():
        param.requires_grad = False
