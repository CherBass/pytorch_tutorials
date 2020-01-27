
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import torchvision
import scipy.misc
import imageio
import pickle
import imgaug as ia
import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib

class ImgAugTransform:
    def __init__(self, gauss_noise, flip_lr, elastic):
        self.aug = iaa.Sequential([
            iaa.AdditiveGaussianNoise(loc=gauss_noise[0], scale=gauss_noise[1]),
            iaa.Fliplr(flip_lr),
            iaa.ElasticTransformation(elastic[0], elastic[1])])

    def __call__(self, image):
        image = np.array(image)
        return self.aug.augment_image(image)

class GenHelper(Dataset):
    def __init__(self, mother, length, mapping):
        # here is a mapping from this index to the mother ds index
        self.mapping=mapping
        self.length=length
        self.mother=mother

    def __getitem__(self, index):
        return self.mother[self.mapping[index]]

    def __len__(self):
        return self.length

def train_valid_split(ds, split_fold=10, random_seed=None):
    '''
    This is a pytorch generic function that takes a data.Dataset object and splits it to validation and training
    efficiently.
    :return: train, val datasets
    '''
    if random_seed!=None:
        np.random.seed(random_seed)

    dslen=len(ds)
    indices= list(range(dslen))
    valid_size=dslen//split_fold
    np.random.shuffle(indices)
    train_mapping=indices[valid_size:]
    valid_mapping=indices[:valid_size]
    train=GenHelper(ds, dslen - valid_size, train_mapping)
    valid=GenHelper(ds, valid_size, valid_mapping)

    return train, valid

class Dataset2D(torch.utils.data.Dataset):
    """2D MRI dataset loader"""

    def __init__(self, root_dir, image_path='',
                 label_path='',
                 transform=None):
        """
        Args:
            root_dir (string): Current directory.
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.transform = transform
        self.root_dir = root_dir
        self.img_path = os.path.join(self.root_dir, image_path)
        file = open(os.path.join(self.root_dir, label_path), 'rb')

        self.labels = pickle.load(file)

        image_paths = sorted(os.listdir(self.img_path))
        remove_ind = []
        i=0
        # check which images are present in labels
        for img in image_paths:
            f = img.split('_')
            f1 = f[5]
            f1 = f1.split('-')
            subject = f1[3]
            if not(any(self.labels['id'].str.match(subject))):
                remove_ind.append(i)
            i=i+1

        # remove images without labels
        image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
        # get unique id list (without duplicates)
        id_list = list(dict.fromkeys(image_paths))
        self.id_list = sorted(id_list)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.id_list)[idx]
        image = np.float32(np.load(os.path.join(self.img_path, img_name)))

        if plot:
            fig = plt.figure()
            a = fig.add_subplot(2, 3, 1)
            imgplot = plt.imshow(image[:,:,0])
            a.axis('off')
            a.set_title('Before aug')
            a = fig.add_subplot(2, 3, 2)
            imgplot = plt.imshow(image[:,:,1])
            a.axis('off')
            a = fig.add_subplot(2, 3, 3)
            imgplot = plt.imshow(image[:,:,2])
            a.axis('off')


        if self.transform:
            image = self.transform(image.astype(np.float16))
            image = image.astype(np.float32)

        if plot:
            a = fig.add_subplot(2, 3, 4)
            imgplot = plt.imshow(image[:,:,0])
            a.axis('off')
            a.set_title('After aug')
            a = fig.add_subplot(2, 3, 5)
            imgplot = plt.imshow(image[:,:,1])
            a.axis('off')
            a = fig.add_subplot(2, 3, 6)
            imgplot = plt.imshow(image[:,:,2])
            a.axis('off')
            plt.show()

        image = torch.from_numpy(image.copy()).float()
        image = image.permute(2,0,1)

        f = img_name.split('_')[5]
        f = f.split('-')
        subject = f[3]
        label = self.labels.loc[self.labels['id'] == subject].iloc[0]
        label = np.array(label['scan_ga'])

        label = torch.from_numpy(label).float()

        return [image, label]
