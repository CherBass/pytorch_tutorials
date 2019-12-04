import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torch.autograd import Variable
import os

class AxonDataset(Dataset):
    """" Inherits pytorch Dataset class to load Axon Dataset """
    def __init__(self, data_name='crops64', folder='', type='train', transform=None):
        """
        :param data_name (string)- data name to load/ save
        :param folder- location of dataset
        :param type - train or test dataset
        """
        self.data_name = data_name
        self.transform = transform
        mmap_mode = 'r'

        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        x_path = os.path.join(__location__, 'npy_data', data_name + '_data_' + type + '.npy')
        y_path = os.path.join(__location__,'npy_data', data_name + '_mask_' + type + '.npy')
        self.x_data = np.load(x_path, mmap_mode=mmap_mode)
        self.y_data = np.load(y_path, mmap_mode=mmap_mode)
        self.len_data = len(self.x_data)

    def __len__(self):
        """ get length of data
        example: len(data) """
        return len(self.x_data)

    def __getitem__(self, idx):
        """gets samples from data according to idx
        :param idx- index to take
        example: data[10] -to get the 10th data sample"""
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        sample_x_data = self.x_data[idx]
        sample_y_data = self.y_data[idx]
        sample_x_data = torch.Tensor(sample_x_data)
        sample_y_data = torch.Tensor(sample_y_data)

        if len(sample_x_data.shape) == 2:
            sample_x_data.unsqueeze_(0)
        if len(sample_y_data.shape) == 2:
            sample_y_data.unsqueeze_(0)

        data = [sample_x_data, sample_y_data]

        return data
