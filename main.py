import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pandas.core.common import flatten
import numpy as np
import matplotlib.pyplot as plt


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
def create_discriminator_model():
    return nn.Sequential(\
        conv3x3(3, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        nn.Flatten(), \
        nn.Linear(64 * 10 * 10, 1))

# test input:
#input = torch.from_numpy(np.random.rand(1,3,160,160).astype(np.float32))

# generator model
def create_generator_model():
    return nn.Sequential(\
        conv3x3(3, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 1, (1,1)), \
        nn.Flatten(), \
        nn.Linear(100, 64*10*10), \
        nn.BatchNorm1d(64*10*10), \
        nn.LeakyReLU(0.2), \
        View((64,10,10)), \
        conv3x3transpose(64,64, (2,2), 1), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3transpose(64,64, (2,2), 0), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3transpose(64,64, (2,2), 0), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3transpose(64,64, (2,2), 0), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 3, 1, 2), \
        Narrow(1), \
        nn.Tanh())
    # result shape: 33x33

# test input:
#input = torch.from_numpy(np.random.rand(20,100).astype(np.float32))
#img  = torch.from_numpy(np.random.rand(20,3,160,160).astype(np.float32))

# generator loss
# gan_model = nn.Sequential(generator_model, discriminator_model)
# gan_model2 = nn.Sequential(generator_model2, discriminator_model2)
# TODO: make descriminator not trainable


# discriminator loss
def create_optimizer(model):
    learning_rate = 0.0002
    return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# learning_rate = 0.0002
# optimizer = torch.optim.Adam(discriminator_model.parameters(), lr=learning_rate, betas=(0.5, 0.999))
# criterion = nn.CrossEntropyLoss()
criterion = nn.BCELoss()


# data loader
transform = transforms.ToTensor()
batch_size = 100

# train_dataset = torchvision.datasets.CIFAR10(root='./data/', \
#                                              train=True, \
#                                              transform=transform, \
#                                              download=True)

# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, \
#                                            batch_size=batch_size, \
#                                            shuffle=True)

image_dataset = torchvision.datasets.ImageFolder(root='./data/resized/')
street_dataset = []
game_dataset = []
for index, (image, label) in enumerate(image_dataset):
    if(label == 0):
        game_dataset = game_dataset + [np.asarray(image).transpose().astype(np.float32) / 128 - 1]
    else:
        street_dataset = street_dataset + [np.asarray(image).transpose().astype(np.float32) / 128 - 1]
street_dataset = np.array(street_dataset)
game_dataset = np.array(game_dataset)

game_discriminator_model = create_discriminator_model()
street_discriminator_model = create_discriminator_model()
game2street_generator_model = create_generator_model()
street2game_generator_model = create_generator_model()

game_discriminator_model.to(device)
street_discriminator_model.to(device)
game2street_generator_model.to(device)
street2game_generator_model.to(device)


# training discriminator
def trainDiscriminator(discriminator_model, realImages, fakeImages):
    discriminator_model = discriminator_model.train()
    for param in discriminator_model.parameters():
        param.requires_grad = True

    realImages = realImages.to(device)
    fakeImages = fakeImages.to(device)
    optimizer = create_optimizer(discriminator_model)
    num_epochs = 10
    for epoch in range(num_epochs):
        outputsReal = discriminator_model(realImages)
        outputsFake = discriminator_model(fakeImages)
        loss = -(torch.mean(outputsReal) - torch.mean(outputsFake))
        print("Training discriminator epoch {} of {}".format(epoch, num_epochs))
        print("loss", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.detach().numpy() < 10**-10:
            print('stop training early as the loss is really small')
            break


def trainCyclicGenerator(forward_generator_model, back_generator_model, discriminator_model, images):
    labels = torch.from_numpy(np.array([1] * len(images)).astype(np.float32))
    images = images.to(device)
    labels = labels.to(device)

    discriminator_model = discriminator_model.eval()
    forward_generator_model = forward_generator_model.train()
    back_generator_model = back_generator_model.eval()
    for param in discriminator_model.parameters():
        param.requires_grad = False
    for param in back_generator_model.parameters():
        param.requires_grad = False
    for param in forward_generator_model.parameters():
        param.requires_grad = True
    cycle_gan_model = nn.Sequential(forward_generator_model, back_generator_model, discriminator_model)
    optimizer = create_optimizer(cycle_gan_model)
    num_epochs = 5
    for epoch in range(num_epochs):
        outputs = cycle_gan_model(images)
        loss = -torch.mean(outputs)
        print("Training cycle_gan_model epoch {} of {}".format(epoch, num_epochs))
        print("loss", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.detach().numpy() < 10**-10:
            print('stop training early as the loss is really small')
            break

def trainGenerator(generator_model, discriminator_model, images):
    labels = torch.from_numpy(np.array([1] * len(images)).astype(np.float32))
    images = images.to(device)
    labels = labels.to(device)
    discriminator_model = discriminator_model.eval()
    generator_model = generator_model.train()
    for param in discriminator_model.parameters():
        param.requires_grad = False
    for param in generator_model.parameters():
        param.requires_grad = True
    gan_model = nn.Sequential(generator_model, discriminator_model)
    optimizer = create_optimizer(gan_model)
    num_epochs = 5
    for epoch in range(num_epochs):
        outputs = gan_model(images)
        loss = -torch.mean(outputs)
        print("Training gan_model epoch {} of {}".format(epoch, num_epochs))
        print("loss", loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if loss.detach().numpy() < 10**-10:
            print('stop training early as the loss is really small')
            break


##### Train Discriminator

game2street_generator_model = game2street_generator_model.eval()
street2game_generator_model = street2game_generator_model.eval()
generated_street_images = game2street_generator_model(torch.from_numpy(game_dataset))
generated_game_images = street2game_generator_model(torch.from_numpy(street_dataset))

trainDiscriminator(game_discriminator_model, torch.from_numpy(game_dataset), generated_game_images)
trainDiscriminator(street_discriminator_model, torch.from_numpy(street_dataset), generated_street_images)

torch.save(game_discriminator_model.state_dict(), './models/game_discriminator.txt')
torch.save(street_discriminator_model.state_dict(), './models/street_discriminator.txt')

## To load the model
# game_discriminator_model.load_state_dict(torch.load('./models/game_discriminator.txt'))
# street_discriminator_model.load_state_dict(torch.load('./models/street_discriminator.txt'))

##### Train Generator

trainCyclicGenerator(game2street_generator_model, street2game_generator_model, game_discriminator_model, torch.from_numpy(game_dataset))
trainCyclicGenerator(street2game_generator_model, game2street_generator_model, street_discriminator_model, torch.from_numpy(street_dataset))

torch.save(game2street_generator_model.state_dict(), './models/game2street_generator.txt')
torch.save(street2game_generator_model.state_dict(), './models/street2game_generator.txt')


def get_shuffle_order_array(length):
    shuffle_order = list(range(0,length))
    np.random.shuffle(shuffle_order)
    return shuffle_order

## To load the model
# game2street_generator_model.load_state_dict(torch.load('./models/game2street_generator.txt'))
# street2game_generator_model.load_state_dict(torch.load('./models/street2game_generator.txt'))


# 0: game dataset; taking game dataset as input; discriminator of game or generated
# 1: street datasets; taking street dataset as input; discriminator of street or generated

def trainingLoop(times = 10):
    for i in range(1, times):
        print("---- round:", i)

        print("- training street generator: ")
        trainGenerator(street2game_generator_model, game_discriminator_model, torch.from_numpy(street_dataset))
        trainCyclicGenerator(game2street_generator_model, street2game_generator_model, game_discriminator_model, torch.from_numpy(game_dataset))
        print("generating images...")
        game2street_generator_model = game2street_generator_model.eval()
        generated_street_images = game2street_generator_model(torch.from_numpy(game_dataset).to(device))
        print("training street discriminator: ")
        trainDiscriminator(street_discriminator_model, torch.from_numpy(np.concatenate([street_dataset, generated_street_images.cpu().detach().numpy()])), torch.from_numpy(np.array([1] * len(street_dataset) + [0] * len(generated_street_images)).astype(np.float32)))

        print("- training game generator: ")
        trainGenerator(game2street_generator_model, street_discriminator_model, torch.from_numpy(game_dataset))
        trainCyclicGenerator(street2game_generator_model, game2street_generator_model, street_discriminator_model, torch.from_numpy(street_dataset))
        print("generating images...")
        street2game_generator_model = street2game_generator_model.eval()
        generated_game_images = street2game_generator_model(torch.from_numpy(street_dataset).to(device))
        print("training game discriminator: ")
        trainDiscriminator(game_discriminator_model, torch.from_numpy(np.concatenate([game_dataset, generated_game_images.cpu().detach().numpy()])), torch.from_numpy(np.array([1] * len(game_dataset) + [0] * len(generated_game_images)).astype(np.float32)))

        torch.save(game2street_generator_model.state_dict(), './models/game2street_generator.txt')
        torch.save(street2game_generator_model.state_dict(), './models/street2game_generator.txt')
        torch.save(game_discriminator_model.state_dict(), './models/game_discriminator.txt')
        torch.save(street_discriminator_model.state_dict(), './models/street_discriminator.txt')


def convertResultAsImage(generated_img):
    return Image.fromarray(((generated_img+1) * 128).cpu().detach().numpy().transpose().astype('uint8'))

def showImage(img):
    fig,ax = plt.subplots()
    ax.imshow(img)
    plt.show()
