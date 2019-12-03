import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch
from torch.autograd import Variable
from load_memmap import *

class AxonDataset(Dataset):
    """" Inherits pytorch Dataset class to load Axon Dataset """
    def __init__(self, data_name='crops64', folder='', type='train', transform=None, resize=None, normalise=False):
        """
        :param data_name (string)- data name to load/ save
        :param folder- location of dataset
        :param type - train or test dataset
        """
        self.data_name = data_name
        self.transform = transform
        self.resize = resize
        self.normalise = normalise

        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        self.x_data, self.y_data = load_dataset(type, folder, data_name)
        self.len_data = len(self.x_data)

    def __len__(self):
        """ get length of data
        example: len(data) """
        return self.len_data

    def __getitem__(self, idx):
        """gets samples from data according to idx
        :param idx- index to take
        example: data[10] -to get the 10th data sample"""
        __location__ = os.path.realpath(
            os.path.join(os.getcwd(), os.path.dirname(__file__)))

        if self.resize:
            sample_x_data = np.resize(np.array([self.x_data[idx]]), (1, self.resize,self.resize))
            sample_y_data = np.resize(np.array([self.y_data[idx]]), (1, self.resize,self.resize))
        else:
            sample_x_data = self.x_data[idx]
            sample_y_data = self.y_data[idx]
        sample_x_data = torch.Tensor(sample_x_data)
        sample_y_data = torch.Tensor(sample_y_data)

        if len(sample_x_data.shape) == 2:
            sample_x_data.unsqueeze_(0)
        if len(sample_y_data.shape) == 2:
            sample_y_data.unsqueeze_(0)

        # normalise between [-1,1]
        if self.normalise:
            sample_x_data = 2*((sample_x_data - torch.min(sample_x_data))/ (torch.max(sample_x_data) - torch.min(sample_x_data)) ) - 1

        data = [sample_x_data, sample_y_data]

        return data
