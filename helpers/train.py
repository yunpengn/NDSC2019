from .network import Cnn_title
import tqdm
import torch.nn as nn
import torch.optim as optimizer
from .parameters import *

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
