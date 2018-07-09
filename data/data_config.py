'''
dataset: 
		PASCAL VOC and SBD
SBD(Semantic Boundary Dataset is a further annotation of the PASCAL VOC data that provides 
more semantic segmentation and instance segmentation masks.) includes 11355 images
https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/dataloaders/custom_transforms.py
'''

from torchvision import transforms 

import torch, cv2

import numpy.random as random
import numpy as np
import math

def fixed_resize(sample, resolution, flagval=None):
    if flagval is None:
        if sample.ndim == 2:
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution, interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution, interpolation=flagval)

    return sample

class FixedResize(object):
    """Resize the image and the ground truth to specified resolution.
    Args:
        size: expected output size of each image
    """
    def __init__(self, size, flagvals=None):
        self.size = (size, size)
        self.flagvals = flagvals

    def __call__(self, sample):


        elems = list(sample.keys())

        for elem in elems:
            if self.flagvals is None:
                sample[elem] = fixed_resize(sample[elem], self.size)
            else:
                sample[elem] = fixed_resize(sample[elem], self.size,
                                                  flagval=self.flagvals[elem])

        return sample

    def __str__(self):
        return 'FixedResize: '+str(self.size)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] /= 255.0
        sample['image'] -= self.mean
        sample['image'] /= self.std

        return sample

class RandomResizedCrop(object):
    """Crop the given Image to random size and aspect ratio.

    A crop of random size (default: of 0.08 to 1.0) of the original size and a random
    aspect ratio (default: of 3/4 to 4/3) of the original aspect ratio is made. This crop
    is finally resized to given size.

    Args:
        size: expected output size of each image
        scale: range of size of the origin size cropped
        ratio: range of aspect ratio of the origin aspect ratio cropped
    """

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.), flagvals=None):
        self.size = (size, size)
        self.scale = scale
        self.ratio = ratio
        self.flagvals = flagvals

    @staticmethod
    def get_params(img, scale, ratio):
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (np.ndarry): Image to be cropped.
            scale (tuple): range of size of the origin size cropped
            ratio (tuple): range of aspect ratio of the origin aspect ratio cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
                sized crop.
        """
        for attempt in range(10):
            area = img.shape[0] * img.shape[1]
            target_area = random.uniform(*scale) * area
            aspect_ratio = random.uniform(*ratio)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w
            if w < img.shape[1] and h < img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return i, j, h, w

        # Fallback
        w = min(img.shape[0], img.shape[1])
        i = (img.shape[0] - w) // 2
        j = (img.shape[1] - w) // 2
        return i, j, w, w

    def __call__(self, sample):
        i, j, h, w = self.get_params(sample['image'], self.scale, self.ratio)
        elems = list(sample.keys())

        for elem in elems:
            if sample[elem].ndim == 2:
                sample[elem] = sample[elem][i:i + h, j:j + w]
            else:
                sample[elem] = sample[elem][i:i + h, j:j + w, :]
            if self.flagvals is None:
                sample[elem] = fixed_resize(sample[elem], self.size)
            else:
                sample[elem] = fixed_resize(sample[elem], self.size,
                                                  flagval=self.flagvals[elem])

        return sample

    def __str__(self):
        return 'RandomResizedCrop: (size={}, scale={}, ratio={}.'.format(str(self.size),
                                                        str(self.scale), str(self.ratio))

class RandomHorizontalFlip(object):
    """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

    def __call__(self, sample):

        if random.random() < 0.5:
            for elem in sample.keys():
                tmp = sample[elem]
                tmp = cv2.flip(tmp, flipCode=1)
                sample[elem] = tmp

        return sample

    def __str__(self):
        return 'RandomHorizontalFlip'


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):

        for elem in sample.keys():
            tmp = sample[elem].astype(np.float32)

            if tmp.ndim == 2:
                tmp = tmp[:, :, np.newaxis]

            # swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W

            tmp = tmp.transpose((2, 0, 1))
            sample[elem] = torch.from_numpy(tmp).float()

        return sample

    def __str__(self):
        return 'ToTensor'

# ------------------------------------------------------------------
INPUT_SIZE = 512

composed_transforms_tr = transforms.Compose([
    RandomResizedCrop(size=INPUT_SIZE, scale=(0.5, 1.0)),
    RandomHorizontalFlip(),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
    ToTensor()])

composed_transforms_ts = transforms.Compose([
    FixedResize(size=INPUT_SIZE),
    Normalize(mean=(0.4914, 0.4822, 0.4465), std=(1.0, 1.0, 1.0)),
    ToTensor()])