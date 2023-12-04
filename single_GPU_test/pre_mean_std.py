from torch.utils.data import DataLoader
from pre_dataset import PreDataset_all

from pre_utils import  list_edf_files_recursive, filter_files_by_size, dict2binary
import torch
import tqdm
import pickle

def get_mean_std(args, files):

    total_sum = 0
    total_samples = 0

    print('####### Start computing mean #######')
    for idx in tqdm.tqdm(range(len(files))):
        if args.data_type == '.edf':
            dataset = PreDataset_all(files[idx], args)

        dataloader = DataLoader(dataset, batch_size=args.train_batch_size)
        device = torch.device('cuda:0' if True else 'cpu')

        for bat_idx, _batch in enumerate(dataloader):
            _batch = _batch.to(device, dtype=torch.float32)
            _batch_samples = _batch.shape[0]
            _batch_sum = torch.sum(_batch, dim=0)

            total_sum += _batch_sum
            total_samples += _batch_samples

    dataset_mean = total_sum / total_samples
    # dataset_mean_add_dim = torch.unsqueeze(dataset_mean, dim=0)

    var_total_sum = 0
    var_total_samples = 0

    print('####### Start computing variance #######')
    for idx in tqdm.tqdm(range(len(files))):
        if args.data_type == '.edf':
            dataset = PreDataset_all(files[idx], args)

        dataloader = DataLoader(dataset, batch_size=args.train_batch_size)
        device = torch.device('cuda:0' if True else 'cpu')

        for bat_idx, _batch in enumerate(dataloader):
            _batch = _batch.to(device, dtype=torch.float32)

            _batch_samples = _batch.shape[0]

            _batch_sum = (_batch - dataset_mean)**2
            _batch_sum = torch.sum(_batch_sum, dim=0)

            var_total_sum += _batch_sum
            var_total_samples += _batch_samples

    dataset_stddev = torch.sqrt(var_total_sum / var_total_samples)

    mean_std_dict = {'mean': dataset_mean,
                     'std': dataset_stddev,
                     'len': var_total_samples}

    dict2binary(f'./tmp/{args.files_len}_mean_std_numch_{args.num_ch}_seqlen_{args.seq_len}.pkl', data_dict=mean_std_dict)


    return dataset_mean, dataset_stddev, var_total_samples