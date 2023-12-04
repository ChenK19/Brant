from torch.utils.data import Dataset
from pre_utils import _load_data, _load_data_edf
from scipy.stats import zscore
import torch
import time
import random

class PreDataset(Dataset):
    def __init__(self, data, power):
        self.power = power
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        res = (self.data[idx],)
        if self.power is not None:
            res += (self.power[idx], )
        return res

class PreDataset_all(Dataset):
    def __init__(self, edf_file_path, args):
        self.edf_file_path = edf_file_path
        self.args = args
        self.data, _ = _load_data_edf(self.edf_file_path, use_power=False, cal_power=False, preload=True, verbose=args.dataloader_verbose)
        self.ch_names = self.data.ch_names
        self.times = self.data.times

        # mean and std initialize as 0,1
        self.mea = 0
        self.st = 1

        self.data = self.data._data

        if self.args.is_normalize:
            time_start =time.time()
            self.data = zscore(self.data, axis=1)
            if args.dataloader_verbose:
                print('Nolmalize time:{}'.format(time.time() - time_start))

        #每一个sample是seq_len*seg_len
        self.num_channel_used = len(self.ch_names) // self.args.num_ch
        self.num_time_used = len(self.times) // (self.args.seq_len * self.args.seg_len)

        self.device = torch.device('cuda:0' if True else 'cpu')


    def __len__(self):
        # 通道上没有重合
        return self.num_channel_used * self.num_time_used

    def __getitem__(self, idx):

        time_start = idx // self.num_channel_used
        ch_start = idx % self.num_channel_used


        res = self.data[ch_start * self.args.num_ch: (ch_start + 1) * self.args.num_ch,
              time_start * (self.args.seq_len * self.args.seg_len): (time_start + 1) * (self.args.seq_len * self.args.seg_len)]

        # seg_num = res.shape[1] // self.args.seg_len
        # board_num = seg_num // self.args.seq_len

        res = torch.tensor(res, dtype=torch.float32)
        # res.to(self.device)

        res = res.view(self.args.num_ch, self.args.seq_len, self.args.seg_len)

        res = (res - self.mea) / self.st

        return res

    def sample_normalize(self, mea, st):
        self.mea = mea
        self.st = st





