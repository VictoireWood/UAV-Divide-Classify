import torch
import cv2
import numpy as np
import os
import glob
from tqdm import trange, tqdm
import platform
import pandas as pd

# import pywt
from PIL import Image

import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, ConcatDataset

import sys
import torch
import logging

import pickle

import he_test
import util
import commons
from classifiers import AAMC, LMCC, LinearLayer, QAMC
from he_datasets import initialize, TrainDataset, TestDataset
from models.helper import GeoClassNet

import parser
from util import get_csv_info

# from generate_database import resolution_h, resolution_w, focal_length


args = parser.parse_arguments()

csv_path = args.csv_path
images_info = get_csv_info(csv_path)

tmp_dir = f'tmp_img/{args.test_dataset_name}'
os.makedirs(tmp_dir, exist_ok=True)

test_transform =T.Compose([
    T.Resize(args.test_resize, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

base_height = 125   # 相当于第一个高度class的中心高度

src_dirs = ['/root/shared-storage/shaoxingyu/workspace_backup/dcqd_test/VPR_h630/', '/root/shared-storage/shaoxingyu/workspace_backup/dcqd_test/VPR_h400/', '/root/shared-storage/shaoxingyu/workspace_backup/dcqd_test/VPR2/']
src_heights = [630, 400, 200]
# dst_dir = '/root/workspace/crikff47v38s73fnfgdg/backup/fake_test_NEW/'
dst_dir = '/root/workspace/crikff47v38s73fnfgdg/backup/fake_test/'

# tmp_save_path = '/root/workspace/fake_test/VPR_h630/'
# input_path = '/root/workspace/dcqddb_test/VPR_h630/@none@630@727927.2284418452@4068986.901224947@.png'

if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


if platform.system() == "Windows":
    slash = '\\'
else:
    slash = '/'

resolution_w = 2048
resolution_h = 1536
# 焦距
focal_length = 1200  # TODO: the intrinsics of the camera

trans_idx_list = [(0, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]   # 分别对应ct,rt,lt,rb,lb
trans_name_list = ['ct', 'rt', 'lt', 'lb', 'rb']

def photo_area_meters(flight_height):
    # 默认width更长
    # # 分辨率
    # resolution_h = 1536
    # resolution_w = 2048
    # # 焦距
    # focal_length = 1200  # TODO: the intrinsics of the camera
    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576
    return map_tile_meters_h, map_tile_meters_w

# def select_pred_utms(pred_utms, pred_height):
#     mid_pred_utm = pred_utms[0]
#     corner_utms = pred_utms[1:]
#     meters_h_pred, meters_w_pred = photo_area_meters(pred_height)
#     corner_line_utm

#     return
    
class group_pred_dataset(torch.utils.data.Dataset):
    def __init__(self, images_info, transform=None):
        """
        初始化数据集
        :param csv_file: 包含图像路径的CSV文件路径
        :param root_dir: 包含图像的根目录
        :param transform: 应用于图像的转换/预处理
        """
        self.images_info = images_info
        # self.root_dir = root_dir
        self.transform = transform
    def __getitem__(self, idx):
        """
        根据索引返回一组图像
        """
        # 获取当前组的图像路径
        tmp_images_paths = [self.images_info[idx][trans_name] for trans_name in trans_name_list]
        # image_paths = self.data_frame.iloc[idx, 1:6].values  # 假设CSV文件的第2列到第6列包含图像路径
        images = []
        for path in tmp_images_paths:
            # 读取图像
            # image = Image.open(os.path.join(self.root_dir, path))
            image = Image.open(path)
            # 应用转换/预处理
            if self.transform:
                image = self.transform(image)
            
            images.append(image)
        images = torch.stack(images, dim = 0)
        pass
        
        # 返回当前组的图像列表
        return images, self.images_info[idx]
    
    def __len__(self):
        """
        返回数据集中的组数
        """
        return len(self.images_info)

def cut_img(images_info, tmp_dir:str):
    cut_bar = tqdm(total=len(images_info))

    for image_info in images_info:

        # 格式image_info = {'image_path': images_paths, 'query_height': query_heights, 'pred_height': pred_class_centers, 'in_threshold': dist <= threshold, 'utm_e', 'utm_n'}
        image_path = image_info['image_path']
        pred_height = image_info['pred_height']
        query_height = image_info['query_height']
        utm_e = image_info['utm_e']
        utm_n = image_info['utm_n']
        origin_meters_h, origin_meters_w = photo_area_meters(query_height)
        zoom_ratio = base_height / pred_height
        move_w = origin_meters_w * (1 - zoom_ratio) / 2
        move_h = origin_meters_h * (1 - zoom_ratio) / 2

        image = cv2.imread(image_path)
        height, width, channels = image.shape
        width = int(width)
        height = int(height)
        
        new_width = round(width * zoom_ratio)
        new_height = round(height * zoom_ratio)

        mid_range = [(height - new_height) // 2, (width - new_width) // 2, (height + new_height) // 2, (width + new_width) // 2]
        trans_w = (width - new_width) // 2
        trans_h = - (height - new_height) // 2

        for i in range(len(trans_idx_list)):
            trans_idx = trans_idx_list[i]
            cropped_utm_gt = (utm_e + trans_idx[0] * move_w, utm_n + trans_idx[1] * move_h)
            
            h1 = max(mid_range[0] + trans_idx[0] * trans_h, 0)
            h2 = min(mid_range[2] + trans_idx[0] * trans_h, height)
            w1 = max(mid_range[1] + trans_idx[1] * trans_w, 0)
            w2 = min(mid_range[3] + trans_idx[1] * trans_w, width)
            new_img = image[h1:h2, w1:w2]
            new_img_resize = cv2.resize(new_img, (width, height), interpolation = cv2.INTER_LANCZOS4)
            filename = os.path.basename(image_path)
            new_filename = f'@{trans_name_list[i]}' + filename
            save_file_path = os.path.join(tmp_dir, new_filename)
            cv2.imwrite(save_file_path, new_img_resize)
            image_info[trans_name_list[i]] = save_file_path
            image_info[trans_name_list[i] + '_utm'] = cropped_utm_gt
            
            # 切取左上角部分
            pass

        cut_bar.update(1)

    return images_info

info_cache_path = f"cache/images_info_{args.test_dataset_name}.pkl"

if os.path.exists(info_cache_path):
    with open(info_cache_path, 'rb') as f:
        images_info = pickle.load(f)
else:
    images_info = cut_img(images_info, tmp_dir)
    with open(info_cache_path, 'wb') as f:
        pickle.dump(images_info, f, pickle.HIGHEST_PROTOCOL)

test_dataset = group_pred_dataset(images_info=images_info, transform=test_transform)
test_dl = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.num_workers>1)



#### he_eval.py
#### Model
if 'dinov2' in args.backbone.lower():
    if args.dinov2_scheme == 'adapter':
        backbone_info = {
            'scheme': 'adapter',
            # 'foundation_model_path': '/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth',
            'foundation_model_path': '/root/shared-storage/shaoxingyu/hub/checkpoints/dinov2_vitb14_pretrain.pth',
            # 'input_size': (210, 280),
            'input_size': args.train_resize,
            # 'input_size': 518,
            # 'input_size': 210,
        }
    elif args.dinov2_scheme == 'finetune':
        if 'salad' in args.aggregator.lower():
            backbone_info={
                'scheme': 'finetune',
                'input_size': args.train_resize,
                'num_trainable_blocks': 4,
                'return_token': True,
                'norm_layer': True,
            }
        else:
            backbone_info = {
                'scheme': 'finetune',
                # 'foundation_model_path': '/root/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth',
                # 'input_size': (210, 280),
                'input_size': args.train_resize,
                'num_trainable_blocks': args.train_blocks_num,
                'return_token': args.return_token,
            }
elif 'efficientnet_v2' in args.backbone.lower():
    backbone_info = {
        # 'input_size': (210, 280),
        'input_size': args.train_resize,
        'layers_to_freeze': 5,
    }
elif 'efficientnet' in args.backbone.lower():
    backbone_info = {
        # 'input_size': (210, 280),
        'input_size': args.train_resize,
        'layers_to_freeze': 5
    }
elif 'radio' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'num_trainable_blocks': args.train_blocks_num,
        'return_token': args.return_token,
        'pre_norm': args.pre_norm,
    }
elif 'resnet' in args.backbone.lower() and 'mixvpr' in args.aggregator.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': 2,
        'layers_to_crop': [4],
    }
elif 'resnet' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': 2,
        'layers_to_crop': [],
    }

if args.aggregator == None:
    agg_config = {}
elif 'resnet50' in args.backbone.lower() and 'mixvpr' in args.aggregator.lower():
    agg_config = {
        'in_channels' : 1024,
        'in_h' : 20,
        'in_w' : 20,
        'out_channels' : 1024,
        'mix_depth' : 4,
        'mlp_ratio' : 1,
        'out_rows' : 4,
    }
elif 'mixvpr' in args.aggregator.lower():
    agg_config = {
        # 'in_channels' : 1280,
        # 'in_h' : 12,
        # 'in_w' : 15,
        'out_channels' : 640,
        'mix_depth' : 4,
        'mlp_ratio' : 1,
        'out_rows' : 4,
    } # the output dim will be (out_rows * out_channels)
elif 'gem' in args.aggregator.lower():
    agg_config={
        'p': 3,
    }
elif 'cosplace' in args.aggregator.lower():
    agg_config={
        # 'in_dim': 
        'out_dim': 2048,
    }
elif 'salad' in args.aggregator.lower():
    agg_config={
        'num_channels': 768,
        'num_clusters': 64,
        'cluster_dim': 128,
        'token_dim': 256,
    }
else:
    agg_config = {}

model = GeoClassNet(args.backbone, backbone_info=backbone_info,aggregator=args.aggregator,agg_config=agg_config).to(args.device)


commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

#### Datasets & DataLoaders
groups = [TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, M=args.M, N=args.N,
                       min_images_per_class=args.min_images_per_class
                       ) for n in range(args.N * args.N)]

# test_dataset = TestDataset(args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize)
# test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

#### Model

# Each group has its own classifier, which depends on the number of classes in the group
if args.classifier_type == "AAMC":
    classifiers = [AAMC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "LMCC":
    classifiers = [LMCC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "FC_CE":
    classifiers = [LinearLayer(model.feature_dim, group.get_classes_num()) for group in groups]
elif args.classifier_type == "QAMC":
    classifiers = [QAMC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]

logging.info(f"Feature dim: {model.feature_dim}")

if args.resume_model is not None:
    model, classifiers = util.resume_model(args, model, classifiers)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

# lr_str, images_info = he_test.inference_he_output(args, model, classifiers, test_dl, groups, images_info)
# logging.info(f"LR: {lr_str}")

lr_str, images_info, ransac_str, ransac_with_weight_str, mid_patch_top1_str, mid_patch_with_weight_str = he_test.inference_he_output_ransac(args, model, classifiers, test_dl, groups, images_info)
logging.info(f'{ransac_str}\n{ransac_with_weight_str}\n{mid_patch_top1_str}\n{mid_patch_with_weight_str}')

result_name = f"{args.dataset_name}_{args.test_dataset_name}_images_info"

new_info_path = os.path.join(args.save_dir, f"{result_name}.pkl")
with open(info_cache_path, 'wb') as f:
    pickle.dump(images_info, f, pickle.HIGHEST_PROTOCOL)
df = pd.DataFrame(images_info)
df.to_csv(os.path.join(args.save_dir, f'{result_name}.csv'), index=False)



#### Validation

