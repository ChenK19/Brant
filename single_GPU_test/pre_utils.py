import os
import random
# import signal

from scipy import signal

import numpy as np
import torch
import torch.distributed as dist

from tqdm import tqdm
import pickle

import mne
import time


def compute_power(data, fs):
    f, Pxx_den = signal.periodogram(data, fs)

    f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
    poses = []
    for fi in range(len(f_thres) - 1):
        cond1_pos = np.where(f_thres[fi] < f)[0]
        cond2_pos = np.where(f_thres[fi + 1] >= f)[0]
        poses.append(np.intersect1d(cond1_pos, cond2_pos))

    ori_shape = Pxx_den.shape[:-1]
    Pxx_den = Pxx_den.reshape(-1, len(f))
    band_sum = [np.sum(Pxx_den[:, band_pos], axis=-1) + 1 for band_pos in poses]
    band_sum = [np.log10(_band_sum)[:, np.newaxis] for _band_sum in band_sum]
    band_sum = np.concatenate(band_sum, axis=-1)
    ori_shape += (8,)
    band_sum = band_sum.reshape(ori_shape)

    return band_sum


def master_save(state_dict, path):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            torch.save(state_dict, path)
    else:
        torch.save(state_dict, path)


def master_print(str):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(str, flush=True)
    else:
        print(str, flush=True)

def binary2dict(binary_file_path):
    with open(binary_file_path, 'rb') as binary_file:
        loaded_data_dict = pickle.load(binary_file)
    return loaded_data_dict

def dict2binary(binary_file_path, data_dict):
    with open(binary_file_path, 'wb') as binary_file:
        pickle.dump(data_dict, binary_file)

def _load_data(file, cal_power, use_power):
    # _data = np.load(os.path.join(file, 'data.npy'))
    _data = binary2dict(file)
    _data = _data['sample']
    _power = None
    if use_power:
        if cal_power:
            _power = compute_power(_data, fs=400)
        else:
            _power = np.load(os.path.join(file, 'power.npy'))

    return _data, _power

def _load_data_edf(edf_file_path, cal_power, use_power, preload=True, verbose=False):

    time_s = time.time()
    _data = mne.io.read_raw_edf(edf_file_path, preload=preload, verbose=verbose)
    if verbose:
        print('load use time: {}'.format(time.time() - time_s))

    return _data, None

def load_data(files, cal_power, use_power=True):
    data, power = [], []
    for file in tqdm(files, disable=True):
        _data, _power = _load_data(file, cal_power=cal_power, use_power=use_power)

        data.append(_data)
        power.append(_power)

    if not use_power:
        power = None
    return data, power

def is_file_smaller_than_1MB(file_path):
    # 获取文件大小，单位为字节
    file_size = os.path.getsize(file_path)

    # 判断是否小于1MB
    return file_size < 1024 * 1024

def filter_files_by_size(file_paths, min_size=1e6):
    # 过滤文件路径，仅保留文件大小大于等于 min_size 的路径
    filtered_paths = [path for path in file_paths if os.path.getsize(path) >= min_size]
    return filtered_paths

def generate_mask(ch_num, seq_len, mask_ratio):
    mask_num = int(ch_num*seq_len*mask_ratio)
    pos = list(range(ch_num*seq_len))

    return random.sample(pos, mask_num)


def get_loss(mask_loss):
    if dist.is_initialized():
        mask_loss_list = [torch.zeros(1).to(dist.get_rank()) for _ in range(dist.get_world_size())]
        dist.all_gather(mask_loss_list, mask_loss)

        return sum(mask_loss_list).item()
    else:
        return mask_loss


def list_edf_files_recursive(directory):
    edf_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".edf"):
                edf_files.append(os.path.join(root, filename))
    return edf_files

def list_pkl_files_recursive(directory):
    pkl_files = []
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".pkl"):
                pkl_files.append(os.path.join(root, filename))
    return pkl_files



