import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms


class GazeDataset(Dataset):
    def __init__(self, transform=None):
        # csvデータの読み出し
        self.transform = transform
        self.dataframe = np.load('../data/Gaze_norm.npy')
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data
    
    
class GazeDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/Gaze_valid_norm.npy')
            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data


class GazeDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/master_experiment/k_ohmori/gaze_eating_data.npy')
            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data