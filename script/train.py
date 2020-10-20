import sys
import os
sys.path.append("..")

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
import argparse
import json

# import VQ-VAE class and Dataset classes
from models.vq_vae import VQ_VAE
from utils.gaze_datasets import GazeDataset
from utils.gaze_datasets import GazeDatasetValid
from utils.gaze_datasets import GazeDatasetTest
from utils.emg_datasets import EMGDataset
from utils.emg_datasets import EMGDatasetValid
from utils.emg_datasets import EMGDatasetTest


def train(model, train_loader, optimizer, writer):
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

def valid(model, valid_loader, writer):
    model.eval()
    recon_error_s = 0
    res_perplexity = 0
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

    return recon_error_s/len(valid_loader), res_perplexity/len(valid_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='gaze', help='gaze or emg')
    parser.add_argument('--f_name', type=str, default='vq_vae', help='.pth name')
    parser.add_argument('--num_hiddens', type=int, default=128, help='parameter of vq-vae encoder and decoder')
    parser.add_argument('--num_residual_hiddens', type=int, default=32, help='parameter of vq-vae')
    parser.add_argument('--num_residual_layers', type=int, default=2, help='the number of residual layer')
    parser.add_argument('--embedding_dim', type=int, default=8, help='embedding dimension')
    parser.add_argument('--num_embeddings', type=int, default=128, help='the number of embeddings')
    parser.add_argument('--commitment_cost', type=float, default=0.25, help='commitment cost')
    parser.add_argument('--decay', type=float, default=0.99, help='decay')
    parser.add_argument('--epoch', type=int, default=500, help='the number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    args = parser.parse_args()

    # Set parameter
    num_hiddens = args.num_hiddens
    num_residual_hiddens = args.num_residual_hiddens
    num_residual_layers = args.num_residual_layers
    embedding_dim = args.embedding_dim
    num_embeddings = args.num_embeddings
    commitment_cost = args.commitment_cost
    decay = args.decay

    batch_size = args.batch_size
    num_epochs = args.epoch
    learning_rate = args.lr


    # Load Data
    if args.data_type == 'gaze':
        dataset = GazeDataset(transform=transforms.ToTensor())
        valid_dataset = GazeDatasetValid(transform=transforms.ToTensor())
        test_dataset = GazeDatasetTest(transform=transforms.ToTensor())
    else:
        dataset = EMGDataset(transform=transforms.ToTensor())
        valid_dataset = EMGDatasetValid(transform=transforms.ToTensor())
        test_dataset = EMGDatasetTest(transform=transforms.ToTensor())

    train_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_dataset, shuffle=False, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define VQ-VAE model
    model = VQ_VAE(num_hiddens, num_residual_layers, num_residual_hiddens,
                num_embeddings, embedding_dim, 
                commitment_cost, decay).to(device)

    # Set log directory and .pth file
    f_name = args.f_name
    name = "pth/{}.pth".format(f_name)
    if not os.path.isdir('pth'):
        os.makedirs('pth')

    log_dir = "logs/{}".format(f_name)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Set 
    metadata = {
        'outputs': [{
            'type': 'tensorboard',
            'source': log_dir,
        }]
    }
    with open('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    # Train
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    train_res_recon_error = []
    train_res_perplexity = []
    recon_error_s = 0
    res_perplexity = 0
    data_variance = 1
    good_id = 0

    for epoch in range(num_epochs):
        train(model, train_loader, optimizer, writer)
        recon_error_s, res_perplexity = valid(model, valid_loader, writer)
        train_res_recon_error.append(recon_error_s)
        train_res_perplexity.append(res_perplexity)

        if recon_error_s <= train_res_recon_error[good_id]:
            torch.save(model.state_dict(), name)
            good_id = epoch

        
    writer.close()