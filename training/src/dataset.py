import torch
from astropy.io import fits
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from astropy.visualization import SqrtStretch, MinMaxInterval
import numpy as np
import h5py

class merger_dataset(Dataset):

    def __init__(self, path_to_file, size, transform = None):
        self.size = size
        self.path_to_file = path_to_file
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, id):

        # Extract dataset from hdf5 file:
        with h5py.File(self.path_to_file, 'r') as hdf:
            density = np.array(hdf['Density'][str(id)]).astype(np.float32)
            density = density[np.newaxis, ...] # To specify the number of channels c (here c=1)
#             MR = np.array(hdf['ratio']['MR'][str(id)]).astype(np.float32)
#             SR = np.array(hdf['ratio']['SR'][str(id)]).astype(np.float32)
            
            ratio = np.array(hdf['Ratio'][str(id)]).astype(np.float32)
        #ratio = np.array([MR,SR])
        sample = {'target': ratio, 'input': density}
        
        if self.transform:
            sample = self.transform(sample)
                   
        return sample


class Normalize(object):  
    def __call__(self, sample):
        ratio, data = sample['target'], sample['input']

        merger_density = minmax(data)

        return {'target': ratio, 'input': merger_density}

    
class ToTensor(object):
    def __call__(self, sample):
        ratio, data = sample['target'], sample['input']

        return {'target': torch.from_numpy(ratio), 'input': torch.from_numpy(data)}

def minmax(array):
    a_min = np.min(array)
    a_max = np.max(array)
    
    if abs(a_max - a_min) > 0:
        array_norm = (array-a_min)/(a_max-a_min)
    else:
        array_norm = np.zeros(array.shape)
    return array_norm 

def splitDataLoader(dataset, split=[0.9, 0.1], batch_size=32, random_seed=None, shuffle=True):
    indices = list(range(len(dataset)))
    s = int(np.floor(split[1] * len(dataset)))
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[s:], indices[:s]

    train_sampler, val_sampler = SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=train_sampler)
    val_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, sampler=val_sampler)

    return train_dataloader, val_dataloader