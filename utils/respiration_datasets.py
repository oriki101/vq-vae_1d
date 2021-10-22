import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

class RespirationDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/respiration_va-vqe_train_data.npy')

        append_data = np.load('../data/respiration_va-vqe_valid_data.npy')
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

class RespirationDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/respiration_va-vqe_valid_data.npy')
        
        append_data = np.load('../data/respiration_va-vqe_train_data.npy')
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

class RespirationNoiseRemovalDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/respiration_va-vqe_train_data_noise_removal.npy')

        append_data = np.load('../data/respiration_va-vqe_valid_data_noise_removal.npy')
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

class RespirationNoiseRemovalDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/respiration_va-vqe_valid_data_noise_removal.npy')
        
        append_data = np.load('../data/respiration_va-vqe_train_data_noise_removal.npy')
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

class RespirationDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.dataframe = np.load('../data/master_experiment/k_ohmori/respiration_eating_data.npy')

            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data