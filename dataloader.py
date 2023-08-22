from typing import Iterator

import random
import torch
from torch.utils.data import Sampler
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, train_images, train_labels) -> None:
        super().__init__()

        self.x_data = train_images
        self.y_data = [[i] for i in train_labels]

    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.LongTensor(self.y_data[idx])

        return x, y
    
class CustomSampler(Sampler):
    def __init__(self, data, pos_num, neg_num):
        self.data = data
        self.total = len(self.data)
        self.pos_num = pos_num
        self.neg_num = neg_num

        self.batch_size = self.pos_num + self.neg_num
        self.iter = self.total // self.batch_size

        self.pos_list_num = len([i for i, sign in enumerate(self.data) if sign != 0])
        self.idx = list(range(self.total))

    def __iter__(self) -> Iterator:
        batch = list()
        for i in range(self.iter):
            temp = np.concatenate((random.sample(self.idx[:self.pos_list_num], self.pos_num),
                random.sample(self.idx[self.pos_list_num:], self.neg_num)))
        random.shuffle(temp)
        batch.extend(temp)

        return iter(batch)
    
    def __len__(self):
        return self.iter * self.batch_size
    
    def get_num_batch(self):
        return self.iter
