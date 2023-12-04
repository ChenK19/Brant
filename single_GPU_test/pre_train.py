import random
import time
import tqdm

import torch
from torch.utils.data import DataLoader
from torch import autocast
from torch.cuda.amp import GradScaler

from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from torch.utils.tensorboard import SummaryWriter

from pre_utils import (generate_mask, master_print, master_save, get_loss, _load_data,
                       list_pkl_files_recursive, list_edf_files_recursive, _load_data_edf, filter_files_by_size, binary2dict)
from pre_dataset import PreDataset, PreDataset_all

import matplotlib.pyplot as plt
from pre_mean_std import get_mean_std

@torch.no_grad()
def log_result(batch, rec, writer, epoch, tot_mask_loss, optimizer):
    if isinstance(batch, list):
        data = batch[0]
    else:
        data = batch

    data_show = data[0]
    rec_show = rec[0]

    ch_num, seq_len, seg_len = data_show.shape

    data_show = data_show.view(ch_num, -1)
    rec_show = rec_show.reshape(ch_num, -1)

    data_show = data_show.cpu().numpy()
    rec_show = rec_show.cpu().numpy()

    fig, axs = plt.subplots(ch_num, 2, figsize=(50, 24))
    for ch in range(ch_num):
        for type in range(2):
            ax = axs[ch, type]
            if type == 0:
                ax.plot(data_show[ch, :])
                ax.set_title(f'ori_data epoch: {epoch}')
            elif type == 1:
                ax.plot(rec_show[ch, :])
                ax.set_title(f'rec_data epoch: {epoch}')

    plt.tight_layout()
    writer.add_figure('recon result', fig, global_step=epoch)

    loss =get_loss(tot_mask_loss)
    writer.add_scalar('Loss/Pre_train', loss, global_step=epoch)

    current_lr = optimizer.param_groups[0]['lr']

    writer.add_scalar('Lr/Pre_train', current_lr, global_step=epoch)




def feed_forward(time_enc, ch_enc,
                 mask, batch, use_power,
                 mask_loss_fn, mask_by_ch, rand_mask, mask_len,
                 rec_down_samp_rate,
                 ):

    if isinstance(batch, list):
        data = batch[0]
    else:
        data = batch

    bat_size, ch_num, seq_len, seg_len = data.shape
    if use_power:
        power = batch[1]
    else:
        power = None
    # time encoder capture long-term dependency
    time_z = time_enc(mask, data, power,
                      need_mask=True,
                      mask_by_ch=mask_by_ch,
                      rand_mask=rand_mask,
                      mask_len=mask_len,
                      use_power=use_power)


    _, _, d_model = time_z.shape
    time_z = time_z.view(bat_size, ch_num, seq_len, d_model)    
    time_z = torch.transpose(time_z, 1, 2)                      
    time_z = time_z.reshape(bat_size*seq_len, ch_num, d_model)  

    _, rec = ch_enc(time_z)             # rec.shape: bat_size*seq_len, ch_num, seg_len // rec_down_samp_rate
    rec = rec.view(bat_size, seq_len, ch_num, seg_len // rec_down_samp_rate)
    rec = torch.transpose(rec, 1, 2)    # transpose back  rec.shape: bat_size, ch_num, seq_len, seg_len
    rec_result = rec
    rec = rec.reshape(bat_size*ch_num*seq_len, seg_len // rec_down_samp_rate)

    data = data.view(bat_size * ch_num * seq_len, seg_len)[::rec_down_samp_rate]
    mask_loss = mask_loss_fn(data, rec)

    return mask_loss, rec_result


def do_epoch(args, epoch,
             time_enc, ch_enc,
             optimizer, scheduler,
             files, scaler, writer, mea, st
             ):
    time_enc.train()
    ch_enc.train()

    mask_loss_fn = torch.nn.MSELoss(reduction='mean')
    tot_mask_loss = 0

    ''' below for pkl training '''
    # for idx in tqdm.tqdm(range(len(files))):
    #     data, power = _load_data(files[idx], cal_power=True, use_power=args.use_power)
    #     # _data, power = _load_data_edf(files[idx], cal_power=True, use_power=args.use_power)
    #
    #     # ch_names = _data.ch_names
    #     data = torch.tensor(data)
    #     ch_num, seg_num, seg_len = data.shape
    #
    #     # truncation and reshape
    #     # 防止不整除
    #     board_num = seg_num // args.seq_len
    #     seg_num = args.seq_len * board_num
    #     pp = data
    #     data = data[:, :seg_num, :].view(ch_num, board_num, args.seq_len, -1)
    #     # data = data[:, :seg_num, :].view(ch_num, board_num, -1, args.seq_len)
    #     data = torch.transpose(data, 0, 1)
    #
    #     if args.use_power:
    #         power = torch.tensor(power)
    #         power = power[:, :seg_num, :].view(ch_num, board_num, args.seq_len, -1)
    #         power = torch.transpose(power, 0, 1)
    #
    #     mask = generate_mask(ch_num, args.seq_len, args.mask_ratio)
    #     dataset = PreDataset(data, power)
    #     bat_size = args.train_batch_size
    #     if dist.is_initialized():
    #         sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=False, drop_last=False)
    #         dataloader = DataLoader(dataset, batch_size=bat_size, num_workers=args.num_workers, drop_last=False, shuffle=False, pin_memory=True, sampler=sampler)
    #         dataloader.sampler.set_epoch(epoch)
    #     else:
    #         dataloader = DataLoader(dataset, batch_size=bat_size, shuffle=False, drop_last=False)
    #
    #     device = torch.device('cuda:0' if True else 'cpu')
    #
    #     for bat_idx, (_batch) in enumerate(dataloader):
    #         _batch[0] = _batch[0].to(device, dtype=torch.float32)
    #         # _batch[0].float()
    #         if args.use_power:
    #             _batch[1] = _batch[1].to(device)
    #             # _batch[1].to(torch.float32)
    #         # with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.amp):
    #         with autocast(device_type='cuda', dtype=torch.float32, enabled=args.amp):
    #             mask_loss = feed_forward(time_enc, ch_enc,
    #                                      mask, _batch, args.use_power,
    #                                      mask_loss_fn,
    #                                      mask_by_ch=args.mask_by_channel,
    #                                      rand_mask=args.rand_mask,
    #                                      mask_len=args.mask_len,
    #                                      rec_down_samp_rate=args.rec_down_samp_rate)
    #             mask_loss = mask_loss / args.accu_step
    #         if args.amp:
    #             scaler.scale(mask_loss).backward()
    #             if (bat_idx+1) % args.accu_step == 0:
    #                 scaler.step(optimizer)
    #                 scaler.update()
    #                 optimizer.zero_grad()
    #                 scheduler.step()
    #         else:
    #             mask_loss.backward()
    #             if (bat_idx + 1) % args.accu_step == 0:
    #                 optimizer.step()
    #                 optimizer.zero_grad()
    #                 scheduler.step()
    #
    #         tot_mask_loss += mask_loss.to(torch.float32)
    #
    # return tot_mask_loss

    ''' below for .edf file training '''
    for idx in tqdm.tqdm(range(len(files))):
        if args.data_type == '.edf':
            dataset = PreDataset_all(files[idx], args)
            dataset.sample_normalize(mea=mea, st=st)

        ch_num = args.num_ch

        mask = generate_mask(ch_num, args.seq_len, args.mask_ratio)

        bat_size = args.train_batch_size

        if dist.is_initialized():
            sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.local_rank, shuffle=False,
                                         drop_last=False)
            dataloader = DataLoader(dataset, batch_size=bat_size, num_workers=args.num_workers, drop_last=False,
                                    shuffle=False, pin_memory=True, sampler=sampler)
            dataloader.sampler.set_epoch(epoch)
        else:
            dataloader = DataLoader(dataset, batch_size=bat_size, shuffle=False, drop_last=False)

        device = torch.device('cuda:0' if True else 'cpu')

        for bat_idx, _batch in enumerate(dataloader):
            _batch = _batch.to(device, dtype=torch.float32)
            with autocast(device_type='cuda', dtype=torch.float32, enabled=args.amp):
                mask_loss, rec_result = feed_forward(time_enc, ch_enc,
                                         mask, _batch, args.use_power,
                                         mask_loss_fn,
                                         mask_by_ch=args.mask_by_channel,
                                         rand_mask=args.rand_mask,
                                         mask_len=args.mask_len,
                                         rec_down_samp_rate=args.rec_down_samp_rate)
                mask_loss = mask_loss / args.accu_step

            # 每个epoch可视化一次结果
            # # if idx == len(files) - 1 and bat_idx == len(dataloader) - 1:
            # if idx == 0 and bat_idx == 0:
            #     log_result(_batch, rec_result, writer, epoch)


            if args.amp:
                scaler.scale(mask_loss).backward()
                if (bat_idx + 1) % args.accu_step == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                mask_loss.backward()
                if (bat_idx + 1) % args.accu_step == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            tot_mask_loss += mask_loss.to(torch.float32)

            # if idx == 0 and bat_idx == 0:
            #     log_result(_batch, rec_result, writer, epoch, tot_mask_loss)



    # 选择最后一份batch做可视化
    log_result(_batch, rec_result, writer, epoch, tot_mask_loss, optimizer)

    return tot_mask_loss


def run_pre_train(args, time_enc, ch_enc, optimizer, scheduler):
    scaler = GradScaler(enabled=args.amp)
    # files = ...  # the data files used for pre-training

    # files = list_pkl_files_recursive(args.data_path)
    if args.data_type == '.edf':
        files = list_edf_files_recursive(args.edf_data_path)
        files = filter_files_by_size(files, min_size=1e6)
        files = files[:args.files_len]

    if args.compute_mean_std:
        mea, st, len_total = get_mean_std(args, files)

    else:
        mean_std_dict = binary2dict(binary_file_path=f'./tmp/{args.files_len}_mean_std_numch_{args.num_ch}_seqlen_{args.seq_len}.pkl')
        # mea, st, len_total = mean_std_dict['mean'], mean_std_dict['std'], mean_std_dict['len']
        mea, st, len_total = mean_std_dict['mean'], mean_std_dict['std'], mean_std_dict['len']

    mea = mea.to('cpu')
    st = st.to('cpu')

    writer = SummaryWriter()

    for epo_idx in range(args.start_epo_idx + 1, args.num_epochs):
        master_print(f'\nEpoch {epo_idx} start')
        start = time.time()
        random.shuffle(files)
        mask_loss = do_epoch(args, epo_idx,
                             time_enc, ch_enc,
                             optimizer, scheduler,
                             files, scaler, writer, mea, st)

        mask_loss = get_loss(mask_loss)
        master_print(f'Train: mask_loss = %.4f' % mask_loss)

        # master_save(time_enc.state_dict(), f'./encoder_ckpt/time_encoder_{args.start_epo_idx}.pt')
        # master_save(ch_enc.state_dict(), f'./encoder_ckpt/channel_encoder_{args.start_epo_idx}.pt')

        if epo_idx % args.model_save_epoch == 0:
            # master_save(time_enc.state_dict(), f'./encoder_ckpt/time_encoder_{epo_idx}_{mask_loss:.6f}.pt')
            # master_save(ch_enc.state_dict(), f'./encoder_ckpt/channel_encoder_{epo_idx}_{mask_loss:.6f}.pt')
            master_save(time_enc.state_dict(), f'{args.model_save_path}time_encoder_{epo_idx}_{mask_loss:.6f}.pt')
            master_save(ch_enc.state_dict(), f'{args.model_save_path}channel_encoder_{epo_idx}_{mask_loss:.6f}.pt')

        master_print('This epoch spends %d s' % (time.time() - start))

    writer.close()

