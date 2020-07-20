from dataset.dataset_retinal import DatasetRetinal
from augmentation.transforms import TransformImgMask, TransformImg
from model.iternet import Unet
from trainer.trainer import Trainer

from argparse import ArgumentParser

def main(args):
    model = Unet(in_channels=3, out_channels=32, kernel_size=(3, 3), do=0.1, num_classes=2)
    trainer = Trainer(gpus=hparams.gpus)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = argparse.ArgumentParser(description='Model Training')
    parser.add_argument('--gpus', default='4', type=str, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--image_dir', default='data/stare/stare-images/', type=str, help='Dataset folder path')
    parser.add_argument('--image_dir', default='data/stare/labels-ah/', type=str, help='Dataset folder path')
    parser.add_argument('--train_csv', default='data/stare/train.csv', type=str, help='list of training set')
    parser.add_argument('--val_csv', default='data/stare/val.csv', type=str, help='list of validation set')
    parser.add_argument('--model_dir', default='', type=str, help='To load pretrained model path')
    parser.add_argument('--lr', default='0.01', type=float, help='learning rate')
    parser.add_argument('--weight_decay', default='0.0001', type=float, help='learning rate')
    parser.add_argument('--n_epochs', default='300', type=int, help='Number of epochs')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--freeze', action='store_true', help='Freeze network except last layer')
    parser.add_argument('--bs', default='16', type=int, help='Batch Size')

    main(args)