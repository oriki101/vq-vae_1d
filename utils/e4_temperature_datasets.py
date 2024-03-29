import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

class TemperatureDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # self.dataframe = np.load('../data/temperature_va-vqe_train_data.npy')

        self.dataframe = np.load('../data/temperature_va-vqe_train_data_moving_average.npy')
        append_data = np.load('../data/temperature_va-vqe_valid_data_moving_average.npy')
        self.dataframe = np.append(self.dataframe, append_data, axis=0)

        # self.dataframe = np.load('../data/temperature_va-vqe_train_data.npy')
        # append_data = np.load('../data/temperature_va-vqe_valid_data.npy')
        # self.dataframe = np.append(self.dataframe, append_data, axis=0)
        
            
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class TemperatureDatasetValid(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # self.dataframe = np.load('../data/temperature_va-vqe_valid_data.npy')
        # self.dataframe = np.load('../data/temperature_va-vqe_valid_data_moving_average.npy')

        self.dataframe = np.load('../data/temperature_va-vqe_train_data_moving_average.npy')
        append_data = np.load('../data/temperature_va-vqe_valid_data_moving_average.npy')
        self.dataframe = np.append(self.dataframe, append_data, axis=0)

        # self.dataframe = np.load('../data/temperature_va-vqe_train_data.npy')
        # append_data = np.load('../data/temperature_va-vqe_valid_data.npy')
        # self.dataframe = np.append(self.dataframe, append_data, axis=0)
            
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data

class TemperatureDatasetTest(Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        # self.dataframe = np.load('../data/temperature_eating_data.npy')
        self.dataframe = np.load('../data/master_experiment/k_ohmori/temperature_eating_data_moving_average.npy')           
    
    # データのサイズ
    def __len__(self):
        return len(self.dataframe)
    
    # データの取得
    def __getitem__(self, idx):
        data = self.dataframe[idx]
        
        if self.transform:
            data = self.transform(data)
        return data