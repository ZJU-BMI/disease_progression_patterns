import random

import pandas as pd
import numpy as np
import torch
from utils import get_normalization
from torch.utils.data import Dataset


class LongitudinalDataset(Dataset):
    def __init__(self, x, y, time, label, attn_mask, time_seq):
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()
        self.time = torch.from_numpy(time).float()
        self.label = torch.from_numpy(label).float()
        self.attn_mask = torch.from_numpy(attn_mask).float()
        self.time_seq = torch.from_numpy(time_seq).float()
        self.len = x.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.time[index], self.label[index], \
               self.attn_mask[index], self.time_seq[index]

    def __len__(self):
        return self.len


def get_attn_mask(seq_lens, max_len=100):
    N = seq_lens.shape[0]
    attn_mask = np.zeros([N, max_len])
    for i in range(N):
        attn_mask[i, :seq_lens[i]] = 1
    return attn_mask


def import_data_adni(data_name='cntomci', normalize=True, normal_mode='normal', data_path=None):
    assert data_name in ['cntomci', 'mcitoad']
    if data_path is None:
        df = pd.read_csv('dataset/ADNI_{}_data.csv'.format(data_name))
    else:
        df = pd.read_csv(data_path)
    if normalize:
        df.iloc[:, 3:] = get_normalization(np.asarray(df.iloc[:, 3:]), normal_mode)
    grouped = df.groupby(['subject_id'])
    features = df.iloc[:, 3:]
    id_list = pd.unique(df['subject_id'])
    N = len(id_list)
    x_dim = features.shape[1]
    # (x1, x2, x3, ..., y), measurements before y
    max_seq_len = np.max(grouped.count())[0] - 1

    data_x = np.zeros([N, max_seq_len, x_dim])
    data_y = np.zeros([N, x_dim])
    time = np.zeros([N], dtype=np.int32)
    label = np.zeros([N], dtype=np.int32)
    seq_lens = np.zeros([N], dtype=np.int32)
    time_seq = np.zeros([N, max_seq_len], dtype=np.int32)
    # get time, label, seq_lens, data_x, data_y
    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)
        len_x = tmp.shape[0] - 1
        # id
        # pat_info[i, 0] = tmp['id'][0]
        # time to progression
        time[i] = np.max(tmp['time'])
        time_seq[i, :len_x] = tmp['time'][0:len_x]
        # indicator of progression
        label[i] = tmp['label'][len_x]
        # number of x
        seq_lens[i] = len_x

        data_x[i, :len_x, :] = tmp.iloc[:len_x, 3:]
        data_y[i, :] = tmp.iloc[len_x, 3:]

    x_dim = data_x.shape[2]
    attn_mask = get_attn_mask(seq_lens, max_len=max_seq_len)

    return (x_dim, max_seq_len), (data_x, data_y), (time, label, attn_mask, time_seq)


def import_longitudinal_data(max_seq_len=5, normalize=True, normal_mode='normal'):
    df = pd.read_csv('dataset/Normalization_mci_data_time_aligned.csv')
    if normalize:
        df.iloc[:, 4:] = get_normalization(np.asarray(df.iloc[:, 4:]), normal_mode)
    grouped = df.groupby(['subject_id'])
    features = df.iloc[:, 4:]
    id_list = pd.unique(df['subject_id'])
    N = len(id_list)
    x_dim = features.shape[1]
    data_x = np.zeros([N, max_seq_len, x_dim])
    time = np.zeros([N], dtype=np.int32)
    label = np.zeros([N, max_seq_len], dtype=np.str_)
    seq_lens = np.zeros([N], dtype=np.int32)
    time_seq = -np.ones([N, max_seq_len], dtype=np.int32)
    ids = np.zeros([N], dtype=np.int32)

    for i, tmp_id in enumerate(id_list):
        tmp = grouped.get_group(tmp_id).reset_index(drop=True)
        len_x = tmp.shape[0]
        ids[i] = tmp_id
        if len_x > 5:
            drop_len = len_x - max_seq_len
            idx = np.arange(1, len_x - 1)
            drop_idx = np.random.choice(idx, drop_len, replace=False)
            tmp.drop(index=drop_idx, inplace=True)

        time_seq[i, :len_x] = tmp['time'][0:len_x]
        data_x[i, :len_x, :] = tmp.iloc[:len_x, 4:]
        label[i, :len_x] = tmp['label'][0:len_x]
        seq_lens[i] = len_x
        # time to progression
        # time[i] = np.max(tmp['time'])
        #
        # # indicator of progression
        # label[i] = tmp['label'][len_x]
        # number of x

    attn_mask = get_attn_mask(seq_lens, max_len=max_seq_len)

    return (x_dim, max_seq_len), data_x, (time, label, attn_mask, time_seq, ids)


def longitudinal_data_to_csv(ids, data_x, time, label, feature_names):
    assert ids.shape[0] == data_x.shape[0]
    assert ids.shape[0] == time.shape[0]
    assert ids.shape[0] == label.shape[0]
    N = len(ids)
    seq_len, x_dim = data_x.shape[1], data_x.shape[2]
    colnames = ['subject_id', 'time', 'label'] + feature_names
    df = pd.DataFrame(columns=colnames, index=np.repeat(np.arange(N), seq_len))
    for i, idx in enumerate(ids):
        df.iloc[i * seq_len: i * seq_len + seq_len, 0] = idx
        df.iloc[i * seq_len: i * seq_len + seq_len, 1] = time[i, :]
        df.iloc[i * seq_len: i * seq_len + seq_len, 2] = label[i, :]
        df.iloc[i * seq_len: i * seq_len + seq_len, 3:] = data_x[i, :, :]

    return df


def preprocess_data(df, normalize=True, normal_mode='normal'):
    if normalize:
        df = get_normalization(np.asarray(df), normal_mode)
    N = len(df)
    x_dim = df.shape[1]
    # (x1, x2, x3, ..., y), measurements before y
    max_seq_len = 1

    data_x = np.asarray(df, dtype=np.float32).reshape([N, max_seq_len, x_dim])
    seq_lens = np.ones([N], dtype=np.int32)
    time_seq = np.zeros([N, 1], dtype=np.int32)
    attn_mask = get_attn_mask(seq_lens, max_len=max_seq_len)
    # return (x_dim, max_seq_len), data_x, (time, label, attn_mask, time_seq)
    return data_x, attn_mask, time_seq


if __name__ == '__main__':
    (x_dim, max_seq_len), data_x, (time, label, attn_mask, time_seq, ids) = import_longitudinal_data()
    df = pd.read_csv('dataset/Normalization_mci_data_time_aligned.csv')
    feature_names = list(df.iloc[:, 4:].columns)
    df = longitudinal_data_to_csv(ids, data_x, time_seq, label, feature_names)
    print()
