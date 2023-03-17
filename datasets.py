import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from utils import get_normalization


class StagingDataset(Dataset):

    def __init__(self, x, y, time, label):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.time = torch.from_numpy(time).float()
        self.label = torch.from_numpy(label).float()
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.time[index], self.label[index]

    def __len__(self):
        return self.len


class StagingDataset2(Dataset):

    def __init__(self, x, y, time, label, pattern):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.time = torch.from_numpy(time).float()
        self.label = torch.from_numpy(label).float()
        self.pattern = torch.from_numpy(pattern).float()
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.time[index], self.label[index], self.pattern[index]

    def __len__(self):
        return self.len


def import_data_ADNI_cntomci(normalize=True, normal_mode='standard'):
    df = pd.read_csv('dataset/ADNI_cntomci_data.csv')
    df_x = df[df['time'] == 0]
    df_y = df[df['time'] > 0]
    # time to event/censoring
    time = np.asarray(df_y['time'])
    # label: 1: AD, 0: MCI
    label = np.asarray(df_y['label'])

    # features
    x = np.asarray(df_x.iloc[:, 3:])
    y = np.asarray(df_y.iloc[:, 3:])

    if normalize:
        x = get_normalization(x, norm_mode=normal_mode)
        y = get_normalization(y, norm_mode=normal_mode)
    DATA = (x, y, time, label)

    return DATA


def import_data_ADNI_mcitoad(normalize=True, normal_mode='standard'):
    df = pd.read_csv('dataset/ADNI_mcitoad_data.csv')
    df_x = df[df['time'] == 0]
    df_y = df[df['time'] > 0]
    # time to event/censoring
    time = np.asarray(df_y['time'])
    # label: 1: AD, 0: MCI
    label = np.asarray(df_y['label'])

    # features
    x = np.asarray(df_x.iloc[:, 3:])
    y = np.asarray(df_y.iloc[:, 3:])

    if normalize:
        x = get_normalization(x, norm_mode=normal_mode)
        y = get_normalization(y, norm_mode=normal_mode)
    DATA = (x, y, time, label)

    return DATA


def import_data_kidney(normalize=True, normal_mode='standard'):
    df = pd.read_csv('dataset/kidney_data_final.csv')
    df_x = df[df['time'] == 0]
    df_y = df[df['time'] > 0]
    # time to event/censoring
    time = np.asarray(df_y['time'])

    label = np.asarray(df_y['label2'])

    # features
    x = np.asarray(df_x.iloc[:, 4:])
    y = np.asarray(df_y.iloc[:, 4:])

    if normalize:
        x = get_normalization(x, norm_mode=normal_mode)
        y = get_normalization(y, norm_mode=normal_mode)
    DATA = (x, y, time, label)

    return DATA


def import_data(dataset='cntomci', normalize=True, normal_mode='standard'):
    if dataset == 'cntomci':
        return import_data_ADNI_cntomci(normalize=normalize, normal_mode=normal_mode)
    elif dataset == 'mcitoad':
        return import_data_ADNI_mcitoad(normalize=normalize, normal_mode=normal_mode)
    elif dataset == 'kidney':
        return import_data_kidney(normalize=normalize, normal_mode=normal_mode)
    else:
        raise NotImplementedError()


