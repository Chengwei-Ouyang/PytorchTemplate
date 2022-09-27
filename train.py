from logging.config import dictConfig
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import os
import hydra
import shutil
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

# model import

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

@hydra.main(version_base=None, config_path='.', config_name='config')
def train(cfg: DictConfig) -> None:
    print('Enter some description here:')
    description = input()
    writer = SummaryWriter(os.path.join(cfg.path.log_path, description))
    shutil.copyfile('config.yaml', os.path.join(cfg.path.log_path, description, 'config.yaml'))

    # declare model, dataset, dataloader, optimizer, loss function

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

    for i in range(cfg.epochs):
        # training
        torch.cuda.empty_cache()
        model.train()

        # validation
        model.eval()
        
        # save model

if __name__ == '__main__':
    train()