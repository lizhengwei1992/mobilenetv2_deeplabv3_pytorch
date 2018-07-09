'''
dataset: 
        PASCAL VOC and SBD
SBD(Semantic Boundary Dataset is a further annotation of the PASCAL VOC data that provides 
more semantic segmentation and instance segmentation masks.) includes 11355 images


Author: Zhengwei Li
Data: July 1 2018
'''


from __future__ import print_function, division
import os

import numpy as np
import scipy.io
import torch.utils.data as data
from PIL import Image
from data import data_config


# -------------------------
# train
#--------------------------

class SBD(data.Dataset):
    """
    SBD(Semantic Boundary Dataset)
    """

    def __init__(self,
                 base_dir='',
                 split='train',
                 transform=True
                 ):
        """
        :param base_dir: path to SBD dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._dataset_dir = os.path.join(self._base_dir, 'dataset')
        self._image_dir = os.path.join(self._dataset_dir, 'img')
        self._cat_dir = os.path.join(self._dataset_dir, 'cls')


        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        # pre_process
        if transform == True:
            self.transform = data_config.composed_transforms_tr


        # Get list of all images from the split and check that the files exist
        self.im_ids = []
        self.images = []
        self.categories = []
        for splt in self.split:
            with open(os.path.join(self._dataset_dir, splt + '.txt'), "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                _image = os.path.join(self._image_dir, line + ".jpg")
                _categ= os.path.join(self._cat_dir, line + ".mat")
                assert os.path.isfile(_image)
                assert os.path.isfile(_categ)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_categ)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images: {:d}'.format(len(self.images)))


    def __getitem__(self, index):

        _img, _target = self._make_img_gt_point_pair(index)

        sample = {'image': _img, 'gt': _target}


        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.images)

    def _make_img_gt_point_pair(self, index):
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        _target = np.array(scipy.io.loadmat(self.categories[index])["GTcls"][0]['Segmentation'][0]).astype(np.float32)

        return _img, _target

# -------------------------
# val
#--------------------------


class VOC(data.Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir='',
                 split='val',
                 transform=True
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        super().__init__()
        self._base_dir = base_dir
        self._image_dir = os.path.join(self._base_dir, 'JPEGImages')
        self._cat_dir = os.path.join(self._base_dir, 'SegmentationClass')

        if isinstance(split, str):
            self.split = [split]
        else:
            split.sort()
            self.split = split
        # pre_process
        if transform == True:
            self.transform = data_config.composed_transforms_ts

        _splits_dir = os.path.join(self._base_dir, 'ImageSets', 'Segmentation')

        self.im_ids = []
        self.images = []
        self.categories = []

        for splt in self.split:
            with open(os.path.join(os.path.join(_splits_dir, splt + '.txt')), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                _image = os.path.join(self._image_dir, line + ".jpg")
                _cat = os.path.join(self._cat_dir, line + ".png")
                assert os.path.isfile(_image)
                assert os.path.isfile(_cat)
                self.im_ids.append(line)
                self.images.append(_image)
                self.categories.append(_cat)

        assert (len(self.images) == len(self.categories))

        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        _img, _target= self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        _img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        _target = np.array(Image.open(self.categories[index])).astype(np.float32)


        return _img, _target
