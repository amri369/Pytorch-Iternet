import os
import torch
import torch.nn as nn
import random
import numpy as np


class Trainer(object):

    def __init__(self, model, criteria, optimizer, scheduler, gpus, seed):
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gpus = gpus
        self.is_gpu_available = torch.cuda.is_available()
        Trainer.set_seed(seed)

    def set_devices(self):
        if self.is_gpu_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            self.model = self.model.cuda()
            self.model = torch.nn.DataParallel(self.model)
            self.criteria = self.criteria.cuda()
        else:
            self.model = self.model.cpu()
            self.criteria = self.criteria.cpu()
            
    def set_seed(seed):
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        random.seed(seed)

        np.random.seed(seed)

        torch.backends.cudnn.deterministic = True

    def training_step(self, dataloader):
        # initialize the loss
        epoch_loss = 0.0

        # loop over training set
        self.model.train()
        for x, y in dataloader:
            if self.is_gpu_available:
                x, y = x.cuda(), y.cuda()

            with torch.set_grad_enabled(True):
                z = self.model(x)
                loss = self.criteria(z[0], y)
                _n = len(z)
                for b in range(1, _n):
                    loss += self.criteria(z[b], y)
                
            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()

            epoch_loss += loss.item() * len(x)

        epoch_loss = epoch_loss / len(dataloader)
        return epoch_loss
    
    def validation_step(self, dataloader):
        # initialize the loss
        epoch_loss = 0.0

        # loop over validation set
        self.model.eval()
        for x, y in dataloader:
            if self.is_gpu_available:
                x, y = x.cuda(), y.cuda()
            with torch.set_grad_enabled(False):
                z = self.model(x)
                loss = self.criteria(z[0], y)
                _n = len(z)
                for b in range(1, _n):
                    loss += self.criteria(z[b], y)

            epoch_loss += loss.item() * len(x)

        epoch_loss = epoch_loss / len(dataloader)
        return epoch_loss

    def save_checkpoint(self, epoch, model_dir):
        # create the state dictionary
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_out_path = model_dir+"_epoch_{}.pth".format(epoch)
        torch.save(state, model_out_path)

    def __call__(self, dataloaders, epochs, model_dir):
        self.set_devices()
        for epoch in range(epochs):
            train_loss = self.training_step(dataloaders['train'])
            val_loss = self.validation_step(dataloaders['val'])
            self.save_checkpoint(epoch, model_dir)
            print('------', epoch+1, '/', epochs, train_loss, val_loss)
            
        if self.is_gpu_available:
                torch.cuda.empty_cache()
