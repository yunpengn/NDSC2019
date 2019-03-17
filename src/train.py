from src.network import Cnn
import tqdm
import torch.nn as nn
import torch.optim as optimizer

model = Cnn()
num_epochs = 2
opt = optimizer.Adam(model.parameters(), lr=1e-2)
loss_func = nn.BCEWithLogitsLoss()


def train(data_loader):
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()

        # Iterates over every data record.
        for x, y in tqdm.tqdm(data_loader):
            opt.zero_grad()
            predication = model(x)
            loss = loss_func(y, predication)
            loss.backward()
            opt.step()

            running_loss += loss.data[0] * x.size(0)

        # Calculates the overall cost for this epoch.
        epoch_loss = running_loss / len(data_loader)
        print('Epoch: {}, training lost: {}'.format(epoch, epoch_loss))
