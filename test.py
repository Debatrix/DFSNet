# coding=utf-8
from argparse import ArgumentParser

import pickle
import numpy as np
import os.path as osp
from PIL import Image
from glob import glob
from tqdm import tqdm
from scipy import stats

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from dataset import monoSimDataset
from model.quality_model import MobileNetV2_Lite


class LoadConfig(object):
    def __init__(self):

        self.mode = 'test'
        self.dataset_path = 'data/cx1'
        self.model_path = "model/pretrained/1211_202056_MobileNetV2_Lite_cx2.pth"
        self.cfg.result_path = ""

        self.seed = 2248

        self.batch_size = 24
        self.device = "cuda:2"
        self.num_workers = 2

        self._change_cfg()

    def _change_cfg(self):
        parser = ArgumentParser()
        for name, value in vars(self).items():
            parser.add_argument('--' + name, type=type(value), default=value)
        args = parser.parse_args()

        for name, value in vars(args).items():
            if self.__dict__[name] != value:
                self.__dict__[name] = value


def test(cfg):
    # cpu or gpu?
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    # data
    print('Loading Data')
    test_data = monoSimDataset(path=cfg.dataset_path,
                               mode='test',
                               seed=cfg.seed,
                               debug_data=False)
    test_data_loader = DataLoader(test_data,
                                  cfg.batch_size,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=cfg.num_workers)

    # configure model
    print('Loading Model')
    model = MobileNetV2_Lite()
    model.to(device)
    if cfg.model_path:
        cp_data = torch.load(cfg.model_path, map_location=device)
        try:
            model.load_state_dict(cp_data['model'])
        except Exception as e:
            model.load_state_dict(cp_data['model'], strict=False)
            print(e)

        cp_data['cfg'] = '' if 'cfg' not in cp_data else cp_data['cfg']
        print(cp_data['cfg'])

    # Start!
    model.eval()
    with torch.no_grad():
        test_pred_loss = 0
        scores = np.zeros((1))
        prediction = np.zeros((1))
        for img, mask, target, _ in tqdm(
                test_data_loader,
                desc='Test',
                bar_format='{desc}: {n_fmt}/{total_fmt} -{percentage:3.0f}%'):
            img = img.to(device)
            target = target.to(device)
            pred, _ = model(img)
            test_pred_loss += nn.functional.mse_loss(pred,
                                                     target,
                                                     reduction='sum')
            scores = np.append(scores, target.cpu().numpy().reshape((-1)))
            prediction = np.append(prediction,
                                   pred.cpu().numpy().reshape((-1)))
        test_pred_loss = test_pred_loss / len(test_data)
        prediction = np.nan_to_num(prediction)
        srocc = stats.spearmanr(prediction[1:], scores[1:])[0]
        lcc = stats.pearsonr(prediction[1:], scores[1:])[0]

        print("Test - MSE: {:.4e}".format(test_pred_loss))
        print("Test - LCC: {:.4f}, SROCC: {:.4f}".format(lcc, srocc))


def predict(cfg):
    # cpu or gpu?
    if torch.cuda.is_available() and cfg.device is not None:
        device = torch.device(cfg.device)
    else:
        if not torch.cuda.is_available():
            print("hey man, buy a GPU!")
        device = torch.device("cpu")

    # data
    img_list = glob(osp.join(cfg.dataset_path, '*.bmp')) + glob(
        osp.join(cfg.dataset_path, '*.png')) + glob(
            osp.join(cfg.dataset_path, '*.jpg'))
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.480], std=[0.200], inplace=False)
    ])

    # configure model
    print('Loading Model')
    model = MobileNetV2_Lite()
    model.to(device)
    if cfg.model_path:
        cp_data = torch.load(cfg.model_path, map_location=device)
        try:
            model.load_state_dict(cp_data['model'])
        except Exception as e:
            model.load_state_dict(cp_data['model'], strict=False)
            print(e)

        cp_data['cfg'] = '' if 'cfg' not in cp_data else cp_data['cfg']
        print(cp_data['cfg'])

    # Start!
    model.eval()
    prediction = {}
    with torch.no_grad():
        for path in tqdm(img_list):
            img_name = osp.basename(path).split('.')[0]
            img = transform(Image.open(path))
            img = img.to(device)
            target = target.to(device)
            pred, heatmap = model(img)
            prediction[img_name] = pred.cpu().numpy().reshape((-1))
            if cfg.result_path:
                heatmap = torch.softmax(heatmap, 0)[1, :, :].cpu().numpy()
                heatmap = Image.fromarray(heatmap)
                heatmap.save(
                    osp.join(cfg.result_path, img_name + '_heatmap.png'))
    if cfg.result_path:
        pickle.dump(prediction, osp.join(cfg.result_path, 'prediction.pkl'))
    return prediction


if __name__ == '__main__':
    cfg = LoadConfig()
    if cfg.mode == 'test':
        test(cfg)
    elif cfg.mode == 'predict':
        prediction = predict(cfg)
