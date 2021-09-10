import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

class BVPDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/bvp_va-vqe_train_data_noise_removal.npy')
        # self.dataframe = self.dataframe.reshape(-1, 1, 3000)
            
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class BVPDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/bvp_va-vqe_valid_data_noise_removal.npy')
        # self.dataframe = self.dataframe.reshape(-1, 1, 3000)
            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class BVPDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/bvp_eating_data.npy')

            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data