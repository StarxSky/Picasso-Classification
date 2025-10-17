import os 
import cv2 
import time
import torch
import numpy as np 
import matplotlib.pyplot as plt 


from torch import nn 
from typing import Any
from torch.backends import mps
from torchvision import models
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter

# Import Model
from model import CustomModel
# Import Dataset
from dataset import TrainDataset

bar = "="
version = torch.__version__
writer = SummaryWriter('./logs')

if mps.is_available() :
    device = torch.device('mps')
elif torch.cuda.is_available() :
    device = torch.device('cuda')
else :
    device = torch.device('cpu')

print(f'{bar*10}Device INFO{bar*10}')
print(f'PyTorch Version :{version}')
print(f'Device :{device}')
print(bar*31)

# Alternative
transformer = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomGrayscale(p=0.5),
    #transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor()
])

#lable_name = ['circle', 'line', 'pentagram', 'quadrilateral', 'rectangle', 'triangle']

dataset = TrainDataset(path='./dataset/train/', transform=transformer) 
#ImageFolder('./train_classified/', transform=transformer)
lable_name = dataset.classes
print(lable_name)

ims, lables = dataset[0]
print(f'DATA: {len(dataset)}')
print(f'Image Shape :{ims.shape}')
print(f'Label :{dataset.vtok[int(lables)]}')


# DataLoader
batch_size = 64
num_works = 0
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_works)
# testing 
for ims, lable in loader :
    print(f'Image Shape :{ims.shape}')
    print(f'Lable :{lable.shape}')
    break


model = CustomModel(num_classes=len(lable_name)).to(device=device)
#print(model)
# set loss function
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Training

def fit(model: nn.Module, loader: DataLoader, optimizer:torch.optim.Optimizer, loss_fn: Any, device: Any, Epochs: int, Loss_List) :
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5)
    print(f'初始化的学习率 :',optimizer.defaults['lr'])

    for epoch in range(Epochs) :
        model.to(device)
        model.train()

        epochs_loss = 0.0
        epochs_acc = 0.0
        step = 0
       
        for image, lable in tqdm(loader, leave=False) :
            image = image.to(device)
            lable = lable.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True) :
                model_out = model(image)
                loss = loss_fn(model_out, lable) # 机算损失
                _, pred = torch.max(model_out, 1)

                loss.backward()
                optimizer.step()
                scheduler.step()

                epochs_acc += torch.sum(pred == lable.data)
                epochs_loss += loss.item() * len(model_out)

                writer.add_scalar(f'{epoch} Acc', epochs_acc, global_step=step)

            step += 1
        
        data_size = len(loader.dataset)
        epochs_loss = epochs_loss / data_size
        epochs_acc = epochs_acc.float() / data_size
        print(f'Epoch {epoch + 1}/{Epochs} | Loss: {epochs_loss:.4f} | Acc: {epochs_acc:.4f}')
        Loss_List.append(epochs_loss)
        
        writer.add_scalar('Loss', epochs_loss, global_step=epoch)
        writer.add_scalar('Acc',epochs_acc, global_step=epoch)

    torch.save(model.state_dict(),'CNN_model.bin')
    print('==>> Mdoel Saved!!')

if __name__ == '__main__':
    epochs_loss_list = []

    fit(model=model,
        loader=loader,
        optimizer=optimizer,
        loss_fn=loss_fn, 
        device=device,
        Epochs=20,
        Loss_List=epochs_loss_list)