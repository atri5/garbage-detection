'''
DataLoader class for performing operations off of the TACO dataset.
Author: Ayush Tripathi (atripathi7783@gmail.com)
'''

import os
import time
import pydicom
from tqdm import tqdm
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import logging

# Set up logger
logging.basicConfig(filename='data_loading_errors.log', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

class TACO_DataLoader(Dataset):
    def __init__(self, image_dir, metadata_dir, transform = None, load_fraction = 1):

        self.image_dir = image_dir
        self.transform = transform
        self.load_fraction = load_fraction
    
        self.coordinates = pd.read_csv(os.path.join(metadata_dir, 'train_label_coordinates.csv'))
        self.metadata = pd.read_csv(os.path.join(metadata_dir, 'train.csv'))
        

        ## TODO: define mappings here ##

        self.data = self.load_data()
    
    def load_data(self):
        pass
    def __getitem__(self, idx):
        pass
    
    def __len__(self):
        return len(self.data)
    
