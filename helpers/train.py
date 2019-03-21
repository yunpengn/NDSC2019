from .network import Cnn_title
import tqdm
import torch.optim as optimizer
from .parameters import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import numpy as np
import json


model = Cnn_title(hidden_dim)
opt = optimizer.Adam(model.parameters(), lr=learning_rate)
loss_func = nn.BCEWithLogitsLoss()


def train(train_loader, val_loader):
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # Iterates over every data record.
        for x, y in tqdm.tqdm(train_loader):
            opt.zero_grad()
            predication = model(x)
            loss = loss_func(y, predication)
            loss.backward()
            opt.step()

            # Updates the running loss.
            running_loss += loss.data * x.size(0)

        # Calculates the overall cost for this epoch.
        epoch_loss = running_loss / len(train_loader)
        print('Epoch: {}, training lost: {}'.format(epoch, epoch_loss))

        # Evaluates the model
        val_loss = 0.0
        model.eval()

        # Iterates over every data record.
        for x, y in val_loader:
            predication = model(x)
            loss = loss_func(y, predication)
            val_loss += loss.data * x.size(0)

        # Calculates the overall cost for this epoch.
        epoch_loss = val_loss / len(val_loader)
        print('Epoch: {}, validation lost: {}'.format(epoch, epoch_loss))


############################### Assume data is preprocessed in data/train and data/test ################

'''
dir_path = 'H:/NDSC-shopee/EXP' ### TODO: change to proper path

data_transform = transforms.Compose([
        transforms.RandomResizedCrop(200),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()#,
#         transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                              std=[0.229, 0.224, 0.225])
    ])

trainset = datasets.ImageFolder(root=dir_path + '/data/train',
                                           transform=data_transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=26, shuffle=True,
                                             num_workers=0)
print("trainloader ready!")
testset = datasets.ImageFolder(root=dir_path+'/data/test',
                                           transform=data_transform)

testloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=26, shuffle=True,
                                             num_workers=0)
print("testloader ready!")

use_gpu = torch.cuda.is_available()
classes = os.listdir(dir_path + '/data/train')
num_of_classes = len(classes)


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training phase
        scheduler.step()
        model.train(True)  # Set model to training mode
        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for data in trainloader:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if 0:  # use_gpu:
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            m = nn.LogSoftmax()
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(m(outputs), labels)

            #             print(m(outputs))
            # forward
            #            outputs = model(inputs)
            #            _, preds = torch.max(outputs.data, 1)
            #            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.data  # [0]
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = int(running_loss) / len(trainset)
            epoch_acc = int(running_corrects) / len(trainset)

            best_acc = max(bast_acc, epoch_acc)
            print(running_corrects)
            print(len(trainset))
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                "train", epoch_loss, epoch_acc))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


# transfer learning resnet18
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, num_of_classes)
# if use_gpu:
#     model_ft = model_ft.cuda()
criterion = nn.NLLLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# train model
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=10)

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model_ft(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy on the test images: %d %%' % (100 * correct / total))
'''
