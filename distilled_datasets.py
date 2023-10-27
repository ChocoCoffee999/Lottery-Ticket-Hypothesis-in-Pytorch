import os
import cv2
import torch
from torch.utils.data import Dataset
#from torchvision import transforms

def load_img(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

class CustomDataset(Dataset):
    def __init__(self, data_path = None, transform= None):
        self.data_path = data_path
        self.transform = transform

        self.images = [os.path.join(data_path, image) for image in os.listdir(data_path)]
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = load_img(self.images[idx])

        if self.transform:
            img = self.transform(img)
        return img
    
    def set_path(self, data_path):
        self.data_path = data_path
        self.images = [os.path.join(data_path, image) for image in os.listdir(data_path)]