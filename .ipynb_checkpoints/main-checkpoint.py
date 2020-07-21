import torch
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from dataset.dataset_retinal import DatasetRetinal
from augmentation.transforms import TransformImg, TransformImgMask
from model.iternet_.iternet_model import Iternet
from trainer.trainer import Trainer

import argparse

def main(args):
    # set the transform
    transform_img_mask = TransformImgMask(
        size=(args.size, args.size), 
        size_crop=(args.crop_size, args.crop_size), 
        to_tensor=True
    )
    
    # set datasets
    csv_dir = {
        'train': args.train_csv,
        'val': args.val_csv
    }
    datasets = {
        x: DatasetRetinal(csv_dir[x], 
                          args.image_dir, 
                          args.mask_dir,
                          batch_size=args.batch_size,
                          transform_img_mask=transform_img_mask, 
                          transform_img=TransformImg()) for x in ['train', 'val']
    }
    
    # set dataloaders
    dataloaders = {
        x: DataLoader(datasets[x], batch_size=args.batch_size, shuffle=True) for x in ['train', 'val']
    }
    
    # initialize the model
    model = Iternet(n_channels=3, n_classes=1, out_channels=32, iterations=3)
    
    # set loss function and optimizer
    criteria = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if model.n_classes > 1 else 'max', patience=2)
    
    # train the model
    trainer = Trainer(model, criteria, optimizer, scheduler, args.gpus, args.seed)
    trainer(dataloaders, args.epochs, args.model_dir)
    torch.cuda.empty_cache()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--gpus', default='4,5,6', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--size', default='592', type=int, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--crop_size', default='128', type=int, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_dir', default='data/stare/stare-images/', type=str, help='Images folder path')
    parser.add_argument('--mask_dir', default='data/stare/labels-ah/', type=str, help='Masks folder path')
    parser.add_argument('--train_csv', default='data/stare/train.csv', type=str, help='list of training set')
    parser.add_argument('--val_csv', default='data/stare/val.csv', type=str, help='list of validation set')
    parser.add_argument('--lr', default='0.001', type=float, help='learning rate')
    parser.add_argument('--epochs', default='2', type=int, help='Number of epochs')
    parser.add_argument('--batch_size', default='32', type=int, help='Batch Size')
    parser.add_argument('--model_dir', default='exp/', type=str, help='Images folder path')
    parser.add_argument('--seed', default='2020123', type=int, help='Random status')
    args = parser.parse_args()
    
    main(args)
    