import sys
sys.path.append("../../src")

import numpy as np
import torch
from torch.utils.data import Dataset

import src.data.interior_dataset as interior_dataset


class MPDataset(Dataset):
    def __init__(self, args, mphase='TRAIN', small_size=False):
        self.mgData = interior_dataset.MGDataset(**args, phase=mphase,
                                                 trainset=("train_fit.txt" if small_size else "train.txt"))

    def __len__(self):
        return self.mgData.__len__()

    def __getitem__(self, index):
        rand_h, rand_w = np.random.randint(0, self.mgData.imHeight), np.random.randint(0, self.mgData.imWidth)
        tmp = {'img': self.mgData[index], 'ref_pos': torch.tensor([rand_h, rand_w])}
        # print(tmp['img']['im'].size(), tmp['ref_pos'].size())
        return tmp
