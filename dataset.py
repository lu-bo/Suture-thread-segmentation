import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from albumentations.pytorch.functional import img_to_tensor
from albumentations import Resize
import random
from scipy import ndimage


class SutureSegDataset(Dataset):
    """docstring for RobotSegDataset"""

    def __init__(self, root, transform, mode, **kwargs):
        super(SutureSegDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.factor = 255
        self.filenames = []
        if mode == 'train':
            for id in range(1, 541):
                if id < 61:
                    img_file = self.root + '/train/imgs/%d.jpg' % id
                # else:
                #     img_file = self.root + '/train/imgs/%d.png' % id
                if id < 61:
                    mask_file = self.root + '/train/masks/%d.png' % id
                else:
                    mask_file = self.root + '/train/masks/%d.png' % 60
                self.filenames.append({'img': img_file, 'label': mask_file})
        else:
            for id in range(1, 21):
                img_file = self.root + '/val/imgs/%d.jpg' % id
                mask_file = self.root + '/val/masks/%d.png' % id
                self.filenames.append({'img': img_file, 'label': mask_file})

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        '''
        pytorch dataloader get_item_from_index
        input:
        idx: corresponding with __len__
        output:
        input_dict: a dictionary stores all return value
        '''
        idx = idx-1
        # input dict for return
        input_dict = {}
        img_file = self.filenames[idx]['img']
        mask_file = self.filenames[idx]['label']


        image = self.load_image(img_file)
        mask = self.load_mask(mask_file)


        # augment
        data = {'image': image, 'mask': mask}

        augmented = self.transform(**data)
        image, mask = augmented["image"], augmented["mask"]

        # input image
        input_dict['input'] = img_to_tensor(image)
        input_dict['target'] = torch.from_numpy(mask).long()
        input_dict['name'] = img_file.split('./')[-1]

        return input_dict



    def load_image(self, filename):
        image = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)

        image = image/255
        return image

    def load_mask(self, filename):
        # change dir name
        mask = cv2.imread(filename, 0)
        mask_bg = (mask / 255)
        mask_fg =  -1*(mask_bg-1)
        return (mask_fg).astype(np.uint8)


    # self defined dataset shuffle
    # random shuffle the index of filenames
    def shuffle_dataset(self):
        self.shuffled_idx = np.arange(0, len(self.filenames))
        np.random.shuffle(self.shuffled_idx)
        self.shuffled_filenames = [self.filenames[idx] for idx in self.shuffled_idx]






import itertools
from torch.utils.data.sampler import Sampler


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)