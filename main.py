import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pandas.core.common import flatten
import numpy as np
import matplotlib.pyplot as plt
import itertools
import time
import datetime


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1, padding=1, bias=False):
    # tensorflow: padding = 'same'
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv3x3transpose(in_channels, out_channels, stride=1, padding=1, bias=False):
    # tensorflow: padding = 'same'
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=padding, bias=bias)

def createResnetBlock(dim):
  return nn.Sequential(\
    conv3x3(dim, dim, (1,1), bias=True), \
    nn.InstanceNorm2d(dim), \
    nn.ReLU(True), \
    conv3x3(dim, dim, (1,1), bias=True), \
    nn.InstanceNorm2d(dim))

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

class Accumulate(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return x + self.model(x)


# discriminator model
def create_discriminator_model():
    return nn.Sequential(\
        conv3x3(3, 64, (2,2)), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 128, (2,2)), \
        nn.BatchNorm2d(128), \
        nn.LeakyReLU(0.2), \
        conv3x3(128, 256, (2,2)), \
        nn.BatchNorm2d(256), \
        nn.LeakyReLU(0.2), \
        conv3x3(256, 512, (2,2)), \
        nn.BatchNorm2d(512), \
        nn.LeakyReLU(0.2), \
        nn.Flatten(), \
        nn.Linear(512 * 10 * 10, 1))

# test input:
#input = torch.from_numpy(np.random.rand(1,3,160,160).astype(np.float32))


# generator model
def create_generator_model():
    return nn.Sequential(\
        conv3x3(3, 64, (2,2), padding=1),
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3(64, 128, (2,2), padding=1), \
        nn.BatchNorm2d(128), \
        nn.LeakyReLU(0.2), \
        conv3x3(128, 256, (2,2), padding=0), \
        nn.BatchNorm2d(256), \
        nn.LeakyReLU(0.2), \
        nn.Sequential(*[Accumulate(createResnetBlock(256)) for i in range(0,9)]), \
        conv3x3transpose(256, 128, (2,2), padding=0), \
        nn.BatchNorm2d(128), \
        nn.LeakyReLU(0.2), \
        conv3x3transpose(128,64, (2,2), padding=0), \
        nn.BatchNorm2d(64), \
        nn.LeakyReLU(0.2), \
        conv3x3transpose(64, 32, (2,2), padding=0), \
        nn.BatchNorm2d(32), \
        nn.LeakyReLU(0.2), \
        conv3x3(32, 3, 1, 2), \
        Narrow(1), \
        nn.Tanh())

# test input:
#input = torch.from_numpy(np.random.rand(20,100).astype(np.float32))
#img  = torch.from_numpy(np.random.rand(20,3,160,160).astype(np.float32))

# generator loss
# gan_model = nn.Sequential(generator_model, discriminator_model)
# gan_model2 = nn.Sequential(generator_model2, discriminator_model2)
# TODO: make descriminator not trainable


# discriminator loss
def create_optimizer(model, override_learning_rate = None):
    learning_rate = override_learning_rate or 0.0002
    return torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.5, 0.999))

# discriminator loss
def create_optimizer_from_multiple_models(models, override_learning_rate = None):
    learning_rate = override_learning_rate or 0.0002
    return torch.optim.Adam(itertools.chain(*map(lambda m: m.parameters(), models)), lr=learning_rate, betas=(0.5, 0.999))


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
        param.data.clamp_(-0.01, 0.01)

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
        del outputsReal, outputsFake, loss

def set_model_for_training(model):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return True

def set_model_for_eval(model):
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return True


def get_generator_loss(generator_A2B, generator_B2A, discriminator_A, discriminator_B, images_A, images_B, lambda_a=10, lambda_b=10):
    map(lambda m: set_model_for_training(m), [generator_A2B, generator_B2A])
    map(lambda m: set_model_for_eval(m), [discriminator_A, discriminator_B])

    cycle_A2A = nn.Sequential(generator_A2B, generator_B2A, discriminator_A) * lambda_a
    cycle_B2B = nn.Sequential(generator_B2A, generator_A2B, discriminator_B) * lambda_b
    gan_A2B = nn.Sequential(generator_A2B, discriminator_B)
    gan_B2A = nn.Sequential(generator_B2A, discriminator_A)

    return sum(map(lambda r: -torch.mean(r), [cycle_A2A(images_A), cycle_B2B(images_B), gan_B2A(images_B) + gan_B2A(images_A)]))


def trainCyclicGeneratorTogether(generator_A2B, generator_B2A, discriminator_A, discriminator_B, images_A, images_B, lambda_a=10, lambda_b=10, override_learning_rate = None):
    optimizer = create_optimizer_from_multiple_models([generator_A2B, generator_B2A], override_learning_rate)

    loss = get_generator_loss(generator_A2B, generator_B2A, discriminator_A, discriminator_B, images_A, images_B, lambda_a, lambda_b)
    print("Training generator epoch {} of {}".format(epoch, num_epochs))
    print("loss", loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_shuffle_order_array(length):
    shuffle_order = list(range(0,length))
    np.random.shuffle(shuffle_order)
    return shuffle_order

## To load the model
# game2street_generator_model.load_state_dict(torch.load('./models/game2street_generator.txt'))
# street2game_generator_model.load_state_dict(torch.load('./models/street2game_generator.txt'))


# A: game dataset; taking game dataset as input; discriminator of game or generated
# B: street datasets; taking street dataset as input; discriminator of street or generated

def trainingLoop(times = 10):
rounds = 121
    for i in range(1, rounds + 1):
      print("---- round:", i)

      startTime = time.time()

      print("- training generators: ")
      override_learning_rate = 0.0002
      trainCyclicGeneratorTogether(game2street_generator_model, street2game_generator_model, game_discriminator_model, street_discriminator_model, game_dataset_at_device, street_dataset_at_device, override_learning_rate=override_learning_rate)

      print("- generating images...")
      set_model_for_eval(game2street_generator_model)
      set_model_for_eval(street2game_generator_model)
      generated_street_images = game2street_generator_model(game_dataset_at_device)
      generated_game_images = street2game_generator_model(street_dataset_at_device)

      print("- training street discriminator: ")
      trainDiscriminator(street_discriminator_model, street_dataset_at_device, generated_street_images)
      del generated_street_images

      print("training game discriminator: ")
      trainDiscriminator(game_discriminator_model, game_dataset_at_device, generated_game_images)
      del generated_game_images

      hoursPerRun = (time.time() - startTime) / 3600.0
      print("... Remaining hours: ", (rounds - i) * hoursPerRun)
      nextSavingPointInMinutes = int(60* (12-i%12) * hoursPerRun)
      print("... Next saving point: (in minutes)", nextSavingPointInMinutes)

      if True or i % 12 == 0:
        torch.save(game2street_generator_model.state_dict(), save_path + 'models/game2street_generator.txt')
        torch.save(street2game_generator_model.state_dict(), save_path + 'models/street2game_generator.txt')
        torch.save(game_discriminator_model.state_dict(), save_path + 'models/game_discriminator.txt')
        torch.save(street_discriminator_model.state_dict(), save_path + 'models/street_discriminator.txt')



def convertResultAsImage(generated_img):
    return Image.fromarray(((generated_img+1) * 128).cpu().detach().numpy().transpose().astype('uint8'))

def showImage(img):
    fig,ax = plt.subplots()
    ax.imshow(img)
    plt.show()
