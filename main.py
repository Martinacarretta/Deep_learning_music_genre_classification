import os
import random
import wandb
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from utils.utils import *

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def model_pipeline(cfg: dict) -> None:
    # Start a new W&B run to track this experiment
    with wandb.init(project="Name", config=cfg):
        config = wandb.config # Get configurtaion from W&B

        dataset = pd.read_csv(config['dataset'])
        print(dataset['labels'].unique())

        # Convert labels to numeric values
        label_mapping = {label: idx for idx, label in enumerate(dataset['labels'].unique())}
        dataset['labels'] = dataset['labels'].map(label_mapping)
        
        print("GETTING STARTED")
        model, train_loader, val_loader, test_loader, criterion, optimizer = make(config, dataset)

        print('PREPARING FOR TRAINING')
        train(model, train_loader, val_loader, criterion, optimizer, config)

        print('TRAINING DONE')
        print('FINALLY, TIME FOR TESTING')
        test(model, test_loader)

    return model

if __name__ == "__main__":
    wandb.login()

    config = dict(
        epochs=10,  #25
        classes=8,
        kernels=[32, 64, 128],
        batch_size=64, 
        learning_rate=1e-4,
        dataset='MY_DATA.csv',
        architecture="ImprovedMusicGenreCNNv2",
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    model = model_pipeline(config)