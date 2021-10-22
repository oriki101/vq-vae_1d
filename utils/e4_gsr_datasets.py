import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
import torch

class GSRDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/gsr_va-vqe_train_data.npy')
        # self.dataframe = self.dataframe.reshape(-1, 1, 3000)

        append_data = np.load('../data/gsr_va-vqe_valid_data.npy')
        self.dataframe = np.append(self.dataframe, append_data, axis=0)


    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class GSRDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/gsr_va-vqe_valid_data.npy')
        # self.dataframe = self.dataframe.reshape(-1, 1, 3000)

        append_data = np.load('../data/gsr_va-vqe_train_data.npy')
        self.dataframe = np.append(self.dataframe, append_data, axis=0)

            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class GSRDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/master_experiment/k_ohmori/gsr_eating_data.npy')

            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data