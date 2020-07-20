from torch.utils.data import Dataset
import torch
import os
import pandas as pd  
from PIL import Image

class DatasetRetinal(Dataset):
    
    def __init__(self, csv_file, image_dir, mask_dir, col_filename='filename', transform_img=None, transform_img_mask=None):
        """
        Args:
            csv_file (string): Path to the csv file with list of images in the dataset.
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the maskes.
            transform_img (callable, optional): Optional transform to be applied on images only.
            transform_img_mask (callable, optional): Optional transform to be applied on images and maskes simultaneously.
        """
        self.filenames = pd.read_csv(csv_file)[col_filename].tolist()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform_img_mask = transform_img_mask
        self.transform_img = transform_img
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # get image data
        filename = self.filenames[idx]
        img_name = os.path.join(self.image_dir, filename)
        img = Image.open(img_name).convert('RGB')
        
        # get mask_data
        mask_name = os.path.join(self.mask_dir, filename)
        mask = Image.open(mask_name).convert('RGB')
        
        if self.transform_img is not None:
            img = self.transform_img(img)
            
        if self.transform_img_mask is not None:
            img, mask = self.transform_img_mask(img, mask)
        
        return img, mask