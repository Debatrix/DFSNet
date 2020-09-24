# coding=utf-8
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import monoSimDataset
from model.loss import FocalLoss, CrossEntropy2d
from model.quality_model import MobileNetV2_Lite


class LoadConfig(object):
    def __init__(self):
        self.info = ""

        self.dataset_path = 'data/cx2'
        self.cp_path = "checkpoints/1203_202301_MobileNetV2_Lite/421_1.3395e-03.pth"
        self.cp_num = 5
        self.visible = True

        self.model = 'MobileNetV2_Lite'
        self.seed = 2248
        self.debug = False
        self.mask_learn_rate = 0.5
        self.mask_lr_decay = 0.1

        self.batch_size = 24
        self.device = "cuda:2"
        self.num_workers = 2

        self.max_epochs = 150
        self.lr = 4e-4
        self.momentum = 0.9
        self.weight_decay = 5e-4

        self._change_cfg()

    def _change_cfg(self):
        parser = ArgumentParser()
        for name, value in vars(self).items():
            parser.add_argument('--' + name, type=type(value), default=value)
        args = parser.parse_args()

        for name, value in vars(args).items():
            if self.__dict__[name] != value:
                self.__dict__[name] = value

        if self.debug:
            self.cp_num = 0
            self.visible = False

    def __str__(self):
        config = ""
        for name, value in vars(self).items():
            config += ('%s=%s\n' % (name, value))
        return config


def train(cfg):
    # configure train
    train_name = time.strftime("%m%d_%H%M%S", time.localtime(
    )) + '_' + cfg.model + '_' + os.path.basename(cfg.dataset_path)
    cfg.name = train_name
    log_interval = int(np.ceil(cfg.max_epochs * 0.1))
    print(cfg)

    # cpu or gpu?
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    # data
    print('Loading Data')
    train_data = monoSimDataset(path=cfg.dataset_path,
                                mode='train',
                                seed=cfg.seed,
                                debug_data=cfg.debug)
    train_data_loader = DataLoader(train_data,
                                   cfg.batch_size,
                                   drop_last=True,
                                   shuffle=True,
                                   num_workers=cfg.num_workers)
    val_data = monoSimDataset(path=cfg.dataset_path,
                              mode='val',
                              seed=cfg.seed,
                              debug_data=cfg.debug)
    val_data_loader = DataLoader(val_data,
                                 cfg.batch_size,
                                 shuffle=False,
                                 drop_last=True,
                                 num_workers=cfg.num_workers)

    # configure model
    print('Loading Model')
    model = MobileNetV2_Lite(True, cfg.mask_learn_rate)
    assert model is not None
    model.to(device)
    if cfg.cp_path:
        cp_data = torch.load(cfg.cp_path, map_location=device)
        try:
            model.load_state_dict(cp_data['model'])
        except Exception as e:
            model.load_state_dict(cp_data['model'], strict=False)
            print(e)

        cp_data['cfg'] = '' if 'cfg' not in cp_data else cp_data['cfg']
        print(cp_data['cfg'])

    # criterion and optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr,
        # momentum=cfg.momentum,
        weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           'min',
                                                           factor=0.5,
                                                           verbose=True)

    pred_criterion = nn.MSELoss()
    mask_criterion = CrossEntropy2d()

    # checkpoint
    if cfg.cp_num > 0:
        cp_dir_path = os.path.normcase(os.path.join('checkpoints', train_name))
        os.mkdir(cp_dir_path)
        best_cp = []
        history_dir_path = os.path.normcase(
            os.path.join(cp_dir_path, 'history'))
        os.mkdir(history_dir_path)
        with open(os.path.normcase(os.path.join(cp_dir_path, 'config.txt')),
                  'w') as f:
            info = str(cfg) + '#' * 30 + '\npre_cfg:\n' + str(
                cp_data['cfg']) if cfg.cp_path else str(cfg)
            f.write(info)

    # visble
    if cfg.visible:
        log_writer = SummaryWriter(os.path.join("log", train_name))
        log_writer.add_text('cur_cfg', cfg.__str__())
        if cfg.cp_path:
            log_writer.add_text('pre_cfg', cp_data['cfg'].__str__())

    # Start!
    print("Start training!\n")
    for epoch in range(1, cfg.max_epochs + 1):
        if epoch % int(cfg.max_epochs / 10) == 0 and cfg.mask_lr_decay < 1:
            cfg.mask_learn_rate *= cfg.mask_lr_decay
            print("[{}] Mask learn rate: {:.4e}".format(
                epoch, cfg.mask_learn_rate))

        # train
        model.train()
        epoch_loss = 0
        for img, mask, target in tqdm(
                train_data_loader,
                desc='[{}] mini_batch'.format(epoch),
                bar_format='{desc}: {n_fmt}/{total_fmt} -{percentage:3.0f}%'):
            img = img.to(device)
            mask = mask.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            pred, heatmap = model(img)
            if cfg.mask_learn_rate == 0:
                loss = pred_criterion(pred, target)
            elif cfg.mask_learn_rate == 0:
                loss = mask_criterion(heatmap, mask)
            else:
                loss = (1 - cfg.mask_learn_rate) * pred_criterion(
                    pred, target) + cfg.mask_learn_rate * mask_criterion(
                        heatmap, mask)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = epoch_loss / len(train_data_loader)
        scheduler.step(train_loss)

        print("[{}] Training - loss: {:.4e}".format(epoch, train_loss))
        if cfg.visible:
            log_writer.add_scalar('Train/Loss', train_loss, epoch)
            log_writer.add_scalar('Train/lr', optimizer.param_groups[0]['lr'],
                                  epoch)

        # val
        if epoch % 5 == 0 or cfg.debug:
            if cfg.model.split('_')[0] == 'MobileNetV3':
                model.train()
            else:
                model.eval()
            with torch.no_grad():
                val_pred_loss = 0
                scores = np.zeros((1))
                prediction = np.zeros((1))
                for img, mask, target in tqdm(
                        val_data_loader,
                        desc='[{}] val_batch'.format(epoch),
                        bar_format=
                        '{desc}: {n_fmt}/{total_fmt} -{percentage:3.0f}%'):
                    img = img.to(device)
                    mask = mask.to(device)
                    target = target.to(device)
                    pred, heatmap = model(img)
                    val_pred_loss += nn.functional.mse_loss(pred,
                                                            target,
                                                            reduction='sum')
                    scores = np.append(scores,
                                       target.cpu().numpy().reshape((-1)))
                    prediction = np.append(prediction,
                                           pred.cpu().numpy().reshape((-1)))
                val_pred_loss = val_pred_loss / len(val_data)
                prediction = np.nan_to_num(prediction)
                srocc = stats.spearmanr(prediction[1:], scores[1:])[0]
                lcc = stats.pearsonr(prediction[1:], scores[1:])[0]

                print("[{}] Val - MSE: {:.4e}".format(epoch, val_pred_loss))
                print("[{}] Val - LCC: {:.4f}, SROCC: {:.4f}".format(
                    epoch, lcc, srocc))
                if cfg.visible:
                    idx = np.random.randint(0, mask.shape[0])
                    heatmap_s = torch.softmax(heatmap, 1)[idx, 1, :, :]
                    log_writer.add_scalar('Val/MSE', val_pred_loss, epoch)
                    log_writer.add_scalar('Val/LCC', lcc, epoch)
                    log_writer.add_scalar('Val/SROCC', srocc, epoch)
                    log_writer.add_image('Val/img', img[idx], epoch)
                    log_writer.add_image('Val/mask',
                                         torch.squeeze(mask[idx]),
                                         epoch,
                                         dataformats='HW')
                    log_writer.add_image('Val/heatmap',
                                         torch.squeeze(heatmap_s),
                                         epoch,
                                         dataformats='HW')

        # checkpoint
        if cfg.cp_num > 0:
            # model.cpu()
            cp_name = "{}_{:.4e}.pth".format(epoch, train_loss)

            if epoch < cfg.cp_num + 1:
                best_cp.append([cp_name, train_loss])
                best_cp.sort(key=lambda x: x[1])
                best_cp_path = os.path.normcase(
                    os.path.join(cp_dir_path, cp_name))

                cp_data = dict(
                    cfg=str(cfg),
                    model=model.state_dict(),
                )
                torch.save(cp_data, best_cp_path)
            else:
                if train_loss < best_cp[-1][1]:
                    os.remove(
                        os.path.normcase(
                            os.path.join(cp_dir_path, best_cp[-1][0])))
                    best_cp[-1] = [cp_name, train_loss]
                    best_cp.sort(key=lambda x: x[1])
                    best_cp_path = os.path.normcase(
                        os.path.join(cp_dir_path, cp_name))
                    cp_data = dict(
                        cfg=str(cfg),
                        model=model.state_dict(),
                    )
                    torch.save(cp_data, best_cp_path)

            if ((log_interval > 0) and (epoch % log_interval == 0 or epoch % 100 == 0)) or \
                    (epoch == cfg.max_epochs):
                history_cp_path = os.path.normcase(
                    os.path.join(history_dir_path, cp_name))
                cp_data = dict(
                    cfg=str(cfg),
                    model=model.state_dict(),
                )
                torch.save(cp_data, history_cp_path)

            # model.to(device)

    return model.cpu()


if __name__ == '__main__':
    cfg = LoadConfig()
    model = train(cfg)
