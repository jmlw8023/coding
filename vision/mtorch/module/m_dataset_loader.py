

import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为宋体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 设备判断
device = ('cuda' if torch.cuda.is_available() else 'mps'  if torch.backends.mps.is_available() else 'cpu')

def show_fashionMNIST_data(training_data):
        
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def download_fashionMNIST_data():

    training_data = datasets.FashionMNIST(
        root='data',
        train=True,
        download=True,
        transform=ToTensor()
    )


    test_data = datasets.FashionMNIST(
        root='data',
        train=False,
        download=True,
        transform=ToTensor()
        
    )


class ImageDatasetFashionMNIST(Dataset):
    
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None) -> None:
        super().__init__()
        
        
        self.img_labels = pd.read_csv(annotation_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        
        
    def __len__(self):
        
        return len(self.img_labels)


    def __getitem__(self, index):
        
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[index, 0])
        
        img = read_image(img_path)
        
        label = self.img_labels.iloc[index, 1]
        
        if self.transform:
            
            img = self.transform(img)
        
        if self.target_transform:
            label = self.target_transform(label)
            
        return img, label
    
    


def train_data(train_data, test_data):
    
    
    train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_dataloader  = DataLoader(test_data, batch_size=64, shuffle=True)







if __name__ == '__main__':
    
    
    
    download_fashionMNIST_data()
    
    
    pass



