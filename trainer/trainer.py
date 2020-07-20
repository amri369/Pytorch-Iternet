import os
import torch


class Trainer(object):

    def __init__(self, model, criteria, optimizer, gpus):
        self.model = model
        self.criteria = criteria
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gpus = gpus
        self.is_gpu_available = torch.cuda.is_available():

    def set_devices(self):
        if self.is_gpu_available:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.gpus
            self.model = self.model.cuda()
            device_ids = [i for i in self.gpus]
            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
            self.criteria = self.criteria.cuda()
            self.optimizer = self.optimizer.cuda()
        else:
            self.model = self.model.cpu()
            self.criteria = self.criteria.cpu()
            self.optimizer = self.optimizer.cpu()

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
                loss = self.criteria(x, z)

            # back propagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()

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
                loss = self.criteria(y, z)
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

        for epoch is range(epochs):
            self.training_step(dataloaders['train'])
            self.validation_step(dataloaders['val'])
            self.save_checkpoint(epoch, model_dir)
