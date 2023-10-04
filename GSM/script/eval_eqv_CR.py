import os
import argparse

import torch
from torch.utils.data import DataLoader

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../shape_assembly'))
from config import get_cfg_defaults
from datasets.baseline.dataloader_CR import ShapeAssemblyDataset
from models.baseline.network_vnn import ShapeAssemblyNet_vnn
import utils


def eval(conf, network, val_dataloader):

    ckpt = torch.load('vnn-network.pth')
    network.load_state_dict(ckpt)


    network.cuda()

    for epoch in range(1):
        tot = 0
        tot_pa = 0
        tot_t = 0
        tot_r = 0
        tot_gd = 0
        val_batches = enumerate(val_dataloader, 0)
        val_fraction_done = 0.0
        val_batch_ind = -1
        device = torch.device('cuda:0')

        ### train for every batch
        for val_batch_ind, val_batch in val_batches:
            if val_batch_ind % 50 == 0:
                print("*" * 10)
                print(epoch, val_batch_ind)
                print("*" * 10)


            network.train()

            for key in val_batch.keys():
                if key not in ['category_name', 'cut_name', 'shape_id', 'result_id']:
                    val_batch[key] = val_batch[key].to(device)
            with torch.no_grad():
                GD, RMSE_R, RMSE_T, PA = network.validation_metrics(batch_data=val_batch, batch_idx=val_batch_ind, vis_idx=val_batch_ind)
            tot_gd += GD.mean()
            tot_r += RMSE_R.mean()
            tot_t += RMSE_T.mean()
            tot_pa += PA.mean()
            tot += 1
        print(tot_gd / tot)
        print(tot_r / tot)
        print(tot_t / tot)
        print(tot_pa / tot)


def main(cfg):
    # Initialize model
    data_features = ['src_pc', 'src_rot', 'src_trans', 'tgt_pc', 'tgt_rot', 'tgt_trans', 'category_name', 'cut_name', 'shape_id', 'result_id']

    model = ShapeAssemblyNet_vnn(cfg=cfg, data_features=data_features).cuda()

    # Initialize val dataloader
    val_data = ShapeAssemblyDataset(
        data_root_dir=cfg.data.root_dir,
        data_csv_file=cfg.data.val_csv_file,
        data_features=data_features,
        num_points=cfg.data.num_pc_points
    )
    val_data.load_data()

    print('Len of Val Data: ', len(val_data))
    print('Distribution of Val Data\n', str(val_data))
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=cfg.exp.batch_size,
        num_workers=cfg.exp.num_workers,
        # persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        drop_last=True
    )

    all_gpus = list(cfg.gpus)
    if len(all_gpus) == 1:
        torch.cuda.set_device(all_gpus[0])

    # Create checkpoint directory
    if not os.path.exists(cfg.exp.log_dir):
        os.makedirs(cfg.exp.log_dir)
    if not os.path.exists(os.path.join(cfg.exp.log_dir, "tb_logs")):
        os.makedirs(os.path.join(cfg.exp.log_dir, "tb_logs"))
    checkpoint_dir = os.path.join(cfg.exp.log_dir, "ckpts")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    eval(cfg, model, val_loader)

    print("Done Evaluation...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg_file', default='', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    # args.cfg_file = './config/train.yml'
    args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    cfg.gpus = cfg.exp.gpus

    cfg.freeze()
    print(cfg)
    main(cfg)
