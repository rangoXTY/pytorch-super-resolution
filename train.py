import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from models.SRCNN import SRCNN
from models.VDSR import VDSR
from dataLoader.T91 import T91


train_file = './data/91-image_x4.h5'
outputs_dir = './data'
scale = 4 
learning_rate = 1e-3
batch_size = 16
num_epochs = 11
print('*** set parameters ***')


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = VDSR().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

train_dataset = T91(train_file)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

print('*** begin train ***')

for epoch in range(1, num_epochs + 1):
    for img, labels in train_dataloader:

        img = img.to(device)
        labels = labels.to(device)

        out = model(img)

        loss = loss_fn(out, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Epochs: %d, Loss: %f" % (epoch, float(loss)))
