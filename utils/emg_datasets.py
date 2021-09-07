import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

class EMGDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/EMG_100Hz_norm_good.npy')
        self.dataframe = self.dataframe.reshape(-1, 1, 3000)
            
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class EMGDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/EMG_valid_100Hz_norm_good.npy')
        self.dataframe = self.dataframe.reshape(-1, 1, 3000)
            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class EMGDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/emg_eating_data.npy')

            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data