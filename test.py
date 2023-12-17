import random

import torchmetrics

from share import *

import pytorch_lightning as pl
import torch
import os
import argparse
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from mvtecad_dataloader import MVTecDataset
from sgn.model import create_model, load_state_dict
from utils.eval_helper import dump, log_metrics, merge_together, performances
from torch.nn import functional as F
import logging
import timm
from scipy.ndimage import gaussian_filter
import cv2
from utils.util import cal_anomaly_map, log_local, create_logger, setup_seed
from visa_dataloader import VisaDataset

parser = argparse.ArgumentParser(description="DiAD")
parser.add_argument("--resume_path", default='./models/output.ckpt')


args = parser.parse_args()

# Configs
resume_path = args.resume_path

batch_size = 1
logger_freq = 300
learning_rate = 1e-5
only_mid_control = True
evl_dir = "npz_result"
logger = create_logger("global_logger", "log/")

# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('models/diad.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'), strict=False)
model.learning_rate = learning_rate
model.only_mid_control = only_mid_control

# Misc
dataset = MVTecDataset('test')
# test_dataset = VisaDataset('test')
dataloader = DataLoader(dataset, num_workers=8, batch_size=batch_size, shuffle=True)
pretrained_model = timm.create_model("resnet50", pretrained=True, features_only=True)
pretrained_model = pretrained_model.cuda()
pretrained_model.eval()

model.eval()
os.makedirs(evl_dir, exist_ok=True)
with torch.no_grad():
    for input in dataloader:
        input_img = input['jpg']
        input_features = pretrained_model(input_img.cuda())
        model = model.cuda()
        output= model.log_images_test(input)
        images = output
        log_local(images, input["filename"][0])
        output_img = images['samples']
        output_features = pretrained_model(output_img.cuda())
        input_features = input_features[1:4]
        output_features = output_features[1:4]

        # Calculate the anomaly score
        anomaly_map, _ = cal_anomaly_map(input_features, output_features, input_img.shape[-1], amap_mode='a')
        anomaly_map = gaussian_filter(anomaly_map, sigma=5)
        anomaly_map = torch.from_numpy(anomaly_map)
        anomaly_map_prediction = anomaly_map.unsqueeze(dim=0).unsqueeze(dim=1)
        input["mask"] = input["mask"]

        root = os.path.join('log_image/')
        name = input["filename"][0][-7:-4]
        filename_feature = "{}-features.jpg".format(name)
        path_feature = os.path.join(root, input["filename"][0][:-7], filename_feature)
        pred_feature = anomaly_map_prediction.squeeze().detach().cpu().numpy()
        pred_feature = (pred_feature * 255).astype("uint8")
        pred_feature = Image.fromarray(pred_feature, mode='L')
        pred_feature.save(path_feature)

        #Heatmap
        anomaly_map_new = np.round(255 * (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min()))
        anomaly_map_new = anomaly_map_new.cpu().numpy().astype(np.uint8)
        heatmap = cv2.applyColorMap(anomaly_map_new, colormap=cv2.COLORMAP_JET)
        pixel_mean = [0.485, 0.456, 0.406]
        pixel_std = [0.229, 0.224, 0.225]
        pixel_mean = torch.tensor(pixel_mean).unsqueeze(1).unsqueeze(1)  # 3 x 1 x 1
        pixel_std = torch.tensor(pixel_std).unsqueeze(1).unsqueeze(1)
        image = (input_img.squeeze() * pixel_std + pixel_mean) * 255
        image = image.permute(1, 2, 0).to('cpu').numpy().astype('uint8')
        image_copy = image.copy()
        out_heat_map = cv2.addWeighted(heatmap, 0.5, image_copy, 0.5, 0, image_copy)
        heatmap_name = "{}-heatmap.png".format(name)
        cv2.imwrite(root + input["filename"][0][:-7] + heatmap_name, out_heat_map)

        input['pred'] = anomaly_map_prediction
        input["output"] = output_img.cpu()
        input["input"] = input_img.cpu()

        output2 = input
        dump(evl_dir, output2)

evl_metrics = {'auc': [ {'name': 'max'}, {'name': 'pixel'}, {'name': 'pro'}, {'name': 'appx'}, {'name': 'apsp'}, {'name': 'f1px'}, {'name': 'f1sp'}]}
print("Gathering final results ...")
fileinfos, preds, masks = merge_together(evl_dir)
ret_metrics = performances(fileinfos, preds, masks, evl_metrics)
log_metrics(ret_metrics, evl_metrics)
