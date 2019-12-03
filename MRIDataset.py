
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

class Dataset2D(torch.utils.data.Dataset):
    """2D MRI dataset loader"""

    def __init__(self, root_dir, image_path='TRAIN_ga_regression/Images', image_path_2=None,
                 label_path='TRAIN_ga_regression/TRAIN_ga_regressionremove_myelin.pkl',
                 task='regression',
                 num_classes=1,
                 label_type='age',
                 output_id=False,
                 test_subjects=None,
                 data_type = 'train',
                 transform=None):
        """
        Args:
            root_dir (string): Current directory.
            image_path (string): path to image folder.
            label_path (string): path to labels file.
            transform (callable, optional): Optional transforms to be applied on a sample.
        """
        self.data_type = data_type
        self.task = task
        self.num_classes = num_classes
        self.transform = transform
        self.root_dir = root_dir
        self.img_path = os.path.join(self.root_dir, image_path)
        if image_path_2:
            self.img_path_2 = os.path.join(self.root_dir, image_path_2)
        else:
            self.img_path_2 = image_path_2

        self.label_type = label_type
        self.output_id = output_id
        file = open(os.path.join(self.root_dir, label_path), 'rb')

        if label_type == 'age':
            self.labels = pickle.load(file)

            ids = sorted(list(self.labels.loc[:,'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]
                # print(len(self.labels))

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
            self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'

            # print(len(image_paths))
            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
            # print(len(image_paths))


        elif label_type == 'cognitive':
            self.labels = pickle.load(file)
            # print(len(self.labels))

            # drop nans
            self.labels = self.labels[pd.notnull(self.labels['composite_score'])]
            self.labels = self.labels[pd.notnull(self.labels['IMD_score'])]
            self.labels = self.labels[self.labels['IMD_score']!=-998]
            # print(len(self.labels))

            ids = sorted(list(self.labels.loc[:, 'id']))

            dupes = [x for n, x in enumerate(ids) if x in ids[:n]]
            dupes = set(dupes)
            dupes = list(dupes)

            for d in dupes:
                temp = self.labels.index[self.labels['id'] == d].tolist()
                # if regression keep first session, if classification keep last session
                if task == 'regression':
                    temp2 = temp[1:]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

                elif task == 'classification':
                    temp2 = temp[:-1]
                    for t in temp2:
                        self.labels = self.labels.drop([t])

            if task == 'classification':
                # print(len(self.labels))
                self.labels = self.labels[self.labels['scan_ga']>((35-40.4482070460464)/1.9935309236699883)]
                # print(len(self.labels))

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

            self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'

            # print(len(image_paths))
            # remove images without labels
            image_paths = [i for j, i in enumerate(image_paths) if j not in remove_ind]
            # print(len(image_paths))

        self.img_pref = f[0] + '_' + f[1] + '__' + f[3] + '_' + f[4] + '_-subj-'
        # get unique id list (without duplicates)
        if data_type == 'train':
            id_list = []
            for img in image_paths:
                f = img.split('_')
                f1 = f[5]
                sess = f[6]
                f1 = f1.split('-')
                id_list.append('sub-' + f1[3] + '_' + sess)
            id_list = list(dict.fromkeys(id_list))

            # remove test subjects
            if test_subjects:
                id_list = [e for e in id_list if e not in test_subjects]
        elif data_type == 'test':
            id_list = test_subjects


        labels_new = pd.DataFrame()
        # keep only labels with images
        for img in id_list:
            f = img.split('_')
            f1 = f[0]
            f1 = f1.split('-')
            subject = f1[1]
            if any(self.labels['id'].str.match(subject)):
                labels_new = labels_new.append(self.labels.loc[self.labels['id'] == subject])
        self.labels = labels_new


        if label_type == 'age':
            # class weights
            is_perm = self.labels['is_prem'].to_numpy().astype(int)
            unique, counts = np.unique(is_perm, return_counts=True)
            class_weights = np.array([len(is_perm)/(num_classes * counts[0]), len(is_perm)/(num_classes * counts[1])])
            self.class_weights = torch.from_numpy(class_weights).float()

        elif label_type == 'cognitive':
            # class weights
            cognitive_bin = (self.labels['composite_score'] > 100).to_numpy().astype(int)
            unique, counts = np.unique(cognitive_bin, return_counts=True)
            class_weights = np.array([len(cognitive_bin)/(num_classes * counts[0]), len(cognitive_bin)/(num_classes * counts[1])])
            self.class_weights = torch.from_numpy(class_weights).float()

        # print([len(self.labels)])
        # print(len(id_list))
        # # randomly choose test subject
        # len_labels = len(image_paths)
        # indices = list(range(len_labels))
        # np.random.shuffle(indices)
        #
        # test_labels = []
        # for t in range(10):
        #     test_labels.append(image_paths[indices[t]])
        # print(test_labels)

        self.image_paths = sorted(image_paths)
        self.id_list = sorted(id_list)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx, plot=False):
        img_name = sorted(self.id_list)[idx]
        img_name = self.img_pref + img_name
        if self.img_path_2:
            rand_num = (np.random.uniform(0,1,1) > 0.5).astype(int)
            if rand_num == 1:
                image = np.float32(np.load(os.path.join(self.img_path, img_name)))
            elif rand_num == 0:
                image = np.float32(np.load(os.path.join(self.img_path_2, img_name)))
        else:
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

        if self.label_type == 'age':
            f = img_name.split('_')[5]
            f = f.split('-')
            subject = f[3]
            if self.task == 'regression':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                label = np.array(label['scan_ga'])
            elif self.task == 'classification':
                values = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_id = 'sub-' + values['id'] + '_ses-' + str(values['session'])
                values = values['is_prem'].astype(int)
                label = np.zeros(self.num_classes)
                label[values] = 1

            label = torch.from_numpy(label).float()

            if self.output_id:
                label = [label, label_id]

        elif self.label_type == 'cognitive':
            f = img_name.split('_')[5]
            f = f.split('-')
            subject = f[3]
            if self.task == 'regression':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                label_cog = np.array(label['composite_score'])
                label_imd = np.array(label['IMD_score'])
                label_birthga = np.array(label['birth_ga'])
                label_scanga = np.array(label['scan_ga'])
                label_corrected_age = np.array(label['corrected_age'])
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                if self.output_id:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0),
                             label_id]
                else:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0)
                             ]
            elif self.task == 'classification':
                label = self.labels.loc[self.labels['id'] == subject].iloc[0]
                cog_temp = (label['composite_score'] > 100).astype(int)
                label_imd = np.array(label['IMD_score'])
                label_birthga = np.array(label['birth_ga'])
                label_scanga = np.array(label['scan_ga'])
                label_corrected_age = np.array(label['corrected_age'])
                label_cog = np.zeros(self.num_classes)
                label_cog[cog_temp] = 1
                label_id = 'sub-' + label['id'] + '_ses-' + str(label['session'])
                if self.output_id:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0),
                             label_id]
                else:
                    label = [torch.from_numpy(label_cog).float(),
                             torch.from_numpy(label_birthga).float().unsqueeze(0),
                             torch.from_numpy(label_scanga).float().unsqueeze(0),
                             torch.from_numpy(label_corrected_age).float().unsqueeze(0),
                             torch.from_numpy(label_imd).float().unsqueeze(0)]

        return [image, label]
