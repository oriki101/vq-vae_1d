import os
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
from torch.autograd import Variable
import torch.nn.functional as F 
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt

# import VQ-VAE class and Dataset classes
from models.vq_vae import VQ_VAE
from utils.gaze_datasets import GazeDataset
from utils.gaze_datasets import GazeDatasetValid
from utils.gaze_datasets import GazeDatasetTest

# Load parameter
import json
json_open = open("./configurations/setting.json", mode='r')
json_load = json.load(json_open)
json_open.close()
num_hiddens = json_load['num_hiddens']
num_residual_hiddens = json_load['num_residual_hiddens']
num_residual_layers = json_load["num_residual_layers"]
embedding_dim = json_load["embedding_dim"]
num_embeddings = json_load["num_embeddings"]
commitment_cost = json_load["commitment_cost"]
decay = json_load["decay"]

batch_size = 32
num_training_updates = 100
num_epochs = 500
learning_rate = 1e-3


# Load Data
dataset = GazeDataset(transform=transforms.ToTensor())
valid_dataset = GazeDatasetValid(transform=transforms.ToTensor())
test_dataset = GazeDatasetTest(transform=transforms.ToTensor())
train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define VQ-VAE model
model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
              num_embeddings, embedding_dim, 
              commitment_cost, decay).to(device)

# Set log directory and .pth file
f_name = "gaze_vq_vae-256-8dim-2"
name = "gaze_VAE_test/{}.pth".format(f_name)
log_dir = "logs/{}".format(f_name)
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)
writer = SummaryWriter(log_dir=log_dir)

# Train
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
train_res_recon_error = []
train_res_perplexity = []
recon_error_s = 0
res_perplexity = 0
data_variance = 1
good_id = 0

for epoch in range(num_epochs):
    model.train()
    recon_error_s = 0
    res_perplexity = 0
    for data in train_loader:
        data = data.to(device, dtype=torch.float)
        data = data.view(data.size(0), 1, -1)
        optimizer.zero_grad()

        vq_loss, data_recon, perplexity = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance        
        loss = recon_error + vq_loss
        
        loss.backward()
        optimizer.step()
        
        recon_error_s += recon_error.item()
        res_perplexity += perplexity.item()
        
    writer.add_scalar("train-loss", recon_error_s/len(train_loader), epoch)
    writer.add_scalar("train-perplexity", res_perplexity/len(train_loader), epoch)
    
    recon_error_s = 0
    res_perplexity = 0
    #検証データで評価
    model.eval()
    with torch.no_grad():
        for data in valid_loader:
            data = data.to(device, dtype=torch.float)
            data = data.view(data.size(0), 1, -1)
            vq_loss, data_recon, perplexity = model(data)
            recon_error = F.mse_loss(data_recon, data) / data_variance
            recon_error_s += recon_error.item()
            res_perplexity += perplexity.item()

    train_res_recon_error.append(recon_error_s/len(valid_loader))
    train_res_perplexity.append(res_perplexity/len(valid_loader))
    writer.add_scalar("valid-loss", recon_error_s/len(valid_loader), epoch)
    writer.add_scalar("valid-perplexity", res_perplexity/len(valid_loader), epoch)
    if recon_error_s/len(train_loader) <= train_res_recon_error[good_id]:
        torch.save(model.state_dict(), name)
        good_id = epoch

    
writer.close()