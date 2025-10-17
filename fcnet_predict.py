import os
import matplotlib.pyplot as plt 
import torch 
from torchvision import transforms
from dataset import TestDataset
from torch.utils.data import DataLoader
from model import FCNet
from FCNN_train import lable_name

import gc

#device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = FCNet(num_classes=len(lable_name))
model.load_state_dict(torch.load('./FCNet_model.bin'))
#model.to(device)

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_images(model, path='./dataset/test'):
    model.to('cpu')
    model.eval()
    predictions = []
    dataset_t = TestDataset(path, transform=transform)
    images = DataLoader(dataset_t, batch_size=1, shuffle=False)
    with torch.no_grad():
        for img in images:
            img = img.unsqueeze(0).to('cpu')  # Add batch dimension
            output = model(img)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            predictions.append(predicted_class)

    gc.collect()
    return predictions



# Prediction
predictions = predict_images(model=model)
#predictions


num_images = len(predictions)
cols = 3
rows = (num_images + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
axes = axes.flatten() if num_images > 1 else [axes]

dataset_t = TestDataset('./dataset/test', transform=transform)
images = DataLoader(dataset_t, batch_size=1, shuffle=False)

for i, v in enumerate(images):
    #print(v.unsqueeze(0).shape)
    axes[i].imshow(torch.permute(v.squeeze(0), (1,2,0)).cpu())
    axes[i].set_title(f"Predicted: {lable_name[predictions[i]]}")
    axes[i].axis('off')

plt.tight_layout()
plt.show()         