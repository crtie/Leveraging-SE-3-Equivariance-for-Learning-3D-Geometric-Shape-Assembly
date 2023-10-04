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
from tensorboardX import SummaryWriter


def train(conf, network, train_dataloader, val_dataloader):

    network_opt = torch.optim.Adam(network.parameters(), lr=conf.optimizer.lr, weight_decay=conf.optimizer.weight_decay)
    val_num_batch = len(val_dataloader)
    train_num_batch = len(train_dataloader)

    network.cuda()
    train_writer = SummaryWriter(os.path.join(conf.exp.log_dir, "tb_logs", 'train'))
    val_writer = SummaryWriter(os.path.join(conf.exp.log_dir, "tb_logs", 'val'))

    for epoch in range(conf.exp.num_epochs):
        train_batches = enumerate(train_dataloader, 0)
        val_batches = enumerate(val_dataloader, 0)
        val_fraction_done = 0.0
        val_batch_ind = -1
        device = torch.device('cuda:0')

        ### train for every batch
        for train_batch_ind, batch in train_batches:
            if train_batch_ind % 50 == 0:
                print("*" * 10)
                print(epoch, train_batch_ind)
                print("*" * 10)
            train_fraction_done = (train_batch_ind + 1) / train_num_batch
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = True
            # log_console = not conf.no_console_log and (last_train_console_log_step is None or \
            #                                            train_step - last_train_console_log_step >= conf.console_log_interval)
            # if log_console:
            #     last_train_console_log_step = train_step

            # save checkpoint
            if epoch % 10 == 0 and train_batch_ind == 0:
                with torch.no_grad():
            #         utils.printout(None, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp.log_dir, 'ckpts', '%d-network.pth' % epoch), _use_new_zipfile_serialization=False)
            #         torch.save(network_opt.state_dict(),
            #                    os.path.join(conf.exp.log_dir, 'ckpts', '%d-optimizer.pth' % epoch), _use_new_zipfile_serialization=False)
            #         # torch.save(network_lr_scheduler.state_dict(),
            #         #            os.path.join(conf.exp.log_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch), _use_new_zipfile_serialization=False)
            #         # torch.save(train_dataset, os.path.join(conf.exp.log_dir, 'ckpts', '%d-train_dataset.pth' % epoch), _use_new_zipfile_serialization=False)
            #         utils.printout(None, 'DONE')

            # set models to training mode
            network.train()
            for key in batch.keys():
                if key not in ['category_name', 'cut_name', 'shape_id', 'result_id']:
                    batch[key] = batch[key].to(device)

            # forward pass (including logging)
            losses = network.training_step(batch_data=batch, batch_idx=train_batch_ind)
            total_loss = losses["total_loss"]
            # point_loss = losses["point_loss"]
            rot_loss = losses["rot_loss"]
            trans_loss = losses["trans_loss"]
            # recon_loss = losses["recon_loss"]

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            # network_lr_scheduler.step()
            if train_batch_ind % 50 == 0:
                print(total_loss.detach().cpu().numpy())

            step = train_step
            train_writer.add_scalar('total_loss', total_loss.item(), step)
            train_writer.add_scalar('rot_loss', rot_loss.item(), step)
            train_writer.add_scalar('trans_loss', trans_loss.item(), step)
            if conf.model.point_loss is True:
                train_writer.add_scalar('point_loss', point_loss.item(), step)

            # validate one batch
            while val_fraction_done <= train_fraction_done and val_batch_ind + 1 < val_num_batch:
                val_batch_ind, val_batch = next(val_batches)

                val_fraction_done = (val_batch_ind + 1) / val_num_batch
                val_step = (epoch + val_fraction_done) * train_num_batch - 1

                log_console = True

                # set models to evaluation mode
                network.train()

                if epoch % 10 == 0 and val_batch_ind <= 5:
                    vis_idx = (epoch+1) * 100000 + val_batch_ind+1
                else:
                    vis_idx = -1

                for key in val_batch.keys():
                    if key not in ['category_name', 'cut_name', 'shape_id', 'result_id']:
                        val_batch[key] = val_batch[key].to(device)
                with torch.no_grad():
                    losses = network.validation_step(batch_data=val_batch, batch_idx=val_batch_ind, vis_idx=vis_idx)
                total_loss = losses["total_loss"]
                if conf.model.point_loss:
                    point_loss = losses["point_loss"]
                if conf.model.recon_loss:
                    recon_loss = losses["recon_loss"]
                rot_loss = losses["rot_loss"]
                trans_loss = losses["trans_loss"]
                if val_batch_ind % 10 == 0:
                    print("val:", total_loss.detach().cpu().numpy())

                step = train_step
                val_writer.add_scalar('total_loss', total_loss.item(), step)
                val_writer.add_scalar('rot_loss', rot_loss.item(), step)
                val_writer.add_scalar('trans_loss', trans_loss.item(), step)
                if conf.model.recon_loss:
                    val_writer.add_scalar('recon_loss', recon_loss.item(), step)
                if conf.model.point_loss:
                    val_writer.add_scalar('point_loss', point_loss.item(), step)


def main(cfg):
    # Initialize model
    data_features = ['src_pc', 'src_rot', 'src_trans', 'tgt_pc', 'tgt_rot', 'tgt_trans', 'category_name', 'cut_name', 'shape_id', 'result_id']

    model = ShapeAssemblyNet_vnn(cfg=cfg, data_features=data_features).cuda()

    # Initialize train dataloader
    train_data = ShapeAssemblyDataset(
        data_root_dir=cfg.data.root_dir,
        data_csv_file=cfg.data.train_csv_file,
        data_features=data_features,
        num_points=cfg.data.num_pc_points
    )
    train_data.load_data()
    print(train_data[0]['src_rot'])
    print(train_data[1]['src_rot'])
    print('Len of Train Data: ', len(train_data))
    print('Distribution of Train Data\n', str(train_data))
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=cfg.exp.batch_size,
        num_workers=cfg.exp.num_workers,
        # persistent_workers=True,
        pin_memory=True,
        shuffle=True,
        drop_last=False
    )

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
        drop_last=False
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

    train(cfg, model, train_loader, val_loader)


    print("Done training...")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument('--cfg_file', default='', type=str)
    parser.add_argument('--gpus', nargs='+', default=-1, type=int)

    args = parser.parse_args()
    # args.cfg_file = './config/train.yml'
    args.cfg_file = os.path.join('./config', args.cfg_file)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg_file)

    # if args.gpus == -1:
    #     args.gpus = [0, 1, 2, 3]
    cfg.gpus = cfg.exp.gpus

    cfg.freeze()
    print(cfg)
    main(cfg)
