from torch.optim import AdamW, Adam
from torchmetrics import AUROC
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import sklearn
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd


class AltData(pl.LightningDataModule):
    def __init__(self, batch_size:int, mcc, ttp, inc, out, inc_h, out_h, cities, y:np.array, val_split=0.2):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = 2
        self.mcc = torch.tensor(mcc).to(torch.float32)
        self.ttp = torch.tensor(ttp).to(torch.float32)
        self.inc = torch.tensor(inc).to(torch.float32)
        self.out = torch.tensor(out).to(torch.float32)
        self.inc_h = torch.tensor(inc_h).to(torch.float32)
        self.out_h = torch.tensor(out_h).to(torch.float32)
        self.cities = torch.tensor(cities).to(torch.float32)
        self.features = [self.mcc, self.ttp, self.inc, self.out, self.inc_h, self.out_h, self.cities]
        
        self.y = torch.tensor(y).to(torch.float32)
        self.val_split = val_split
        
    
    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            generator = torch.Generator().manual_seed(42)
            dataset = TensorDataset(*self.features, self.y)
            self.train, self.val = random_split(dataset, 
                    [int(len(dataset)*(1 - self.val_split)), int(len(dataset)*self.val_split)], generator=generator)
    
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size)
    
    
    
class AltModel(pl.LightningModule):
    def __init__(self, lr=1e-5, loss=nn.BCEWithLogitsLoss()):
        super().__init__()
        self.lr = lr
        self.loss = loss
        self.save_hyperparameters()
        self.auroc = AUROC(task='binary')
        
        self.inc_mlp = nn.Sequential(
            nn.Linear(184, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
        )
        
        self.out_mlp = nn.Sequential(
            nn.Linear(184, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        self.inch_mlp = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        self.outh_mlp = nn.Sequential(
            nn.Linear(24, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        
        self.mcc_mlp = nn.Sequential(
            nn.Linear(184, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        self.tt_mlp = nn.Sequential(
            nn.Linear(155, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64)
        )
        
        self.city_mlp = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )
        
        
        self.mlp = nn.Sequential(
            nn.Linear(64 * 4 + 32 * 2 + 16 * 1, 128),
            nn.Dropout(p=0.4), 
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.2), 
            nn.Linear(64, 1)
        )
        
    def forward(self, mcc, ttp, inc, out, inc_h, out_h, cities):
        mcc = self.mcc_mlp(mcc)
        ttp = self.tt_mlp(ttp)
        inc = self.inc_mlp(inc)
        out = self.out_mlp(out)
        inc_h = self.inch_mlp(inc_h)
        out_h = self.outh_mlp(out_h)
        cities = self.city_mlp(cities)
        inp = torch.cat([mcc, ttp, inc, out, inc_h, out_h, cities], 1)
        outp = self.mlp(inp)
        return outp
    
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr = self.lr)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.2, cooldown=1)
        return  {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    
    def training_step(self, train_batch, batch_idx):
        mcc, ttp, inc, out, inc_h, out_h, cities, y = train_batch
        logits = self.forward(mcc, ttp, inc, out, inc_h, out_h, cities)
        loss = self.loss(logits, y.view(-1, 1))
        self.log("train_loss", loss)
        y1 = y.clone().to(torch.int32).view(-1, 1)
        self.log("train_auroc", self.auroc(logits, y1))
        return loss
    
    def validation_step(self, valid_batch, batch_idx):
        mcc, ttp, inc, out, inc_h, out_h, cities, y = valid_batch
        logits = self.forward(mcc, ttp, inc, out, inc_h, out_h, cities)
        loss = self.loss(logits, y.view(-1, 1))
        self.log("val_loss", loss)
        y1 = y.clone().to(torch.int32).view(-1, 1)
        self.log("val_auroc", self.auroc(logits, y1))
        