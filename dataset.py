import os
import sys 
import torch
import gc 
from PIL import Image
from torch.utils.data import Dataset
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

class TrainDataset(Dataset):
    def __init__(self, path:str, transform):
        super().__init__()
        self.trans = transform
        self.path = path
        self.dict = {
            'circle': 0,
            'delta': 3,
            'line' : 2,
            'rectangle': 4,
            'star': 5
        }
        self.vtok = dict((v, k) for k, v in self.dict.items())
        self.classes = list(self.dict.keys())
        


    def read_image(self, path_root:str):

        files = os.listdir(path=path_root)
        images, labels = [], []
        for filename in files:
            img_path = os.path.join(self.path, filename)
            image = Image.open(img_path).convert('RGB')
            image = self.trans(image)
            

            # Special
            _dict = {
                'circle' : 0,
                'line' : 2,
                'pentagram': 5, # star
                'quadrilateral': 3,  # delta
                'rectangle': 4,
            }
            
            if '_' in filename and filename.split('_')[1] in list(_dict.keys()) :
                filename_bb = filename.split('_')[1]
                label = torch.tensor(_dict[filename_bb], dtype=torch.int64)
            elif '_' in filename and filename.split('_')[1] in ['0', '2', '3', '5', '4']:
                filename_bb = filename.split('_')[1]
                label = torch.tensor(int(filename_bb), dtype=torch.int64)
            else: 
                filename_bb = filename.split('-')[1]
                label = torch.tensor(int(filename_bb), dtype=torch.int64)



            images.append(image)
            labels.append(label)
        return images, labels
    
    def __len__(self):
        images, _ = self.read_image(self.path)
        gc.collect
        return len(images)
    

    def __getitem__(self, index):

        images_t, labels_t = self.read_image(self.path)
        
        
        X = images_t[index].to(device)
        Y = labels_t[index].to(device)

        return X, Y
    



class TestDataset(Dataset):
    def __init__(self, path:str, transform):
        super().__init__()
        self.trans = transform
        self.path = path
        self.dict = {
            'circle': 0,
            'delta': 3,
            'line' : 2,
            'rectangle': 4,
            'star': 5
        }
        self.vtok = dict((v, k) for v, k in self.dict.items())
        self.classes = list(self.dict.keys())
        


    def read_image(self, path_root:str):

        files = os.listdir(path=path_root)
        images = []
        for filename in files:
            img_path = os.path.join(self.path, filename)
            image = Image.open(img_path).convert('RGB')
            image = self.trans(image)
            images.append(image)
            
                          
        return images
    
    def __len__(self):
        return len(os.listdir(path=self.path))
    
    
    def __getitem__(self, index):

        images_t = self.read_image(self.path)

        X = images_t[index].to(device)
        

        return X
    


if __name__ == '__main__':
    from torchvision import transforms
    transformer = transforms.Compose([
    transforms.Resize([64,64]),
    transforms.RandomHorizontalFlip(p=0.5),
    #transforms.RandomGrayscale(p=0.5),
    #transforms.ColorJitter(saturation=0.5),
    transforms.ToTensor()
    ])
    dataset = TrainDataset(path='./dataset/train/', transform=transformer) 

    lable_name = dataset.classes
    print(lable_name)
    print(dataset[0][0].shape)
    import matplotlib.pyplot as plt 
    plt.imshow(torch.permute(dataset[0][0], (1,2,0)).cpu())
    plt.show()