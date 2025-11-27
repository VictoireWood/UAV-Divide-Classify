import os
import sys
import pickle
import time
import psutil
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

import parser
from classifiers import AAMC, LMCC, LinearLayer, QAMC
from he_datasets import TrainDataset
from models.helper import GeoClassNet
import util
from util import get_csv_info
import commons
import he_test

resolution_w = 2048
resolution_h = 1536
# 焦距
focal_length = 1200  # TODO: the intrinsics of the camera
base_height = 125   # 相当于第一个高度class的中心高度

trans_idx_list = [(0, 0)]   # 分别对应ct,rt,lt,rb,lb
trans_name_list = ['ct']  # 只保留中心块

def cut_img(images_info, tmp_dir:str):
    cut_time_list = []
    memory_usage_list = []  # 新增：记录内存使用
    cut_bar = tqdm(total=len(images_info))
    for image_info in images_info:
        # 获取加载前的内存
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        image_path = image_info['image_path']
        image = cv2.imread(image_path)
        cut_start = time.time()
        pred_height = image_info['pred_height']
        # query_height = image_info['query_height']
        utm_e = image_info['utm_e']
        utm_n = image_info['utm_n']
        
        # origin_meters_h, origin_meters_w = photo_area_meters(query_height)
        zoom_ratio = base_height / pred_height
        height, width, channels = image.shape
        
        new_width = round(width * zoom_ratio)
        new_height = round(height * zoom_ratio)

        # 只处理中心块
        h1 = (height - new_height) // 2
        h2 = (height + new_height) // 2
        w1 = (width - new_width) // 2
        w2 = (width + new_width) // 2
        
        new_img = image[h1:h2, w1:w2]
        new_img_resize = cv2.resize(new_img, (width, height), interpolation=cv2.INTER_LANCZOS4)
        cut_time_list.append(time.time() - cut_start)
        # 获取加载后的内存
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_usage_list.append(mem_after - mem_before)

        filename = os.path.basename(image_path)
        new_filename = f'@ct{filename}'  # ct表示center
        save_file_path = os.path.join(tmp_dir, new_filename)
        cv2.imwrite(save_file_path, new_img_resize)
        
        image_info['ct'] = save_file_path
        image_info['ct_utm'] = (utm_e, utm_n)
        
        cut_bar.update(1)
    return images_info, cut_time_list, memory_usage_list


args = parser.parse_arguments()

info_cache_path = f"cache/images_info_{args.test_dataset_name}.pkl"
tmp_dir = f'tmp_img/{args.test_dataset_name}'
if os.path.exists(info_cache_path):
    avg_cut_time = 0.0
    avg_cpu_crop_mem = 0.0
    with open(info_cache_path, 'rb') as f:
        images_info = pickle.load(f)
else:
    csv_path = args.csv_path
    os.makedirs(tmp_dir, exist_ok=True)
    images_info = get_csv_info(csv_path)
    images_info, cut_time_list, memory_usage_list = cut_img(images_info, tmp_dir)
    avg_cut_time = np.mean(cut_time_list)
    avg_cpu_crop_mem = np.mean(memory_usage_list)
    with open(info_cache_path, 'wb') as f:
        pickle.dump(images_info, f, pickle.HIGHEST_PROTOCOL)

test_transform =T.Compose([
    T.Resize(args.test_resize, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def photo_area_meters(flight_height):
    map_tile_meters_w = resolution_w / focal_length * flight_height   # 相机内参矩阵里focal_length的单位是像素
    map_tile_meters_h = resolution_h / focal_length * flight_height # NOTE w768*h576
    return map_tile_meters_h, map_tile_meters_w
    
class group_pred_dataset(torch.utils.data.Dataset):
    def __init__(self, images_info, transform=None):
        """
        初始化数据集
        :param csv_file: 包含图像路径的CSV文件路径
        :param transform: 应用于图像的转换/预处理
        """
        self.images_info = images_info
        self.transform = transform
    def __getitem__(self, idx):
        """
        根据索引返回一组图像
        """
        # 获取当前组的图像路径
        tmp_images_path = self.images_info[idx]['ct']
        image = Image.open(tmp_images_path)
        # 应用转换/预处理
        if self.transform:
            image = self.transform(image) 
        # 返回当前组的图像列表
        return image, self.images_info[idx]
    
    def __len__(self):
        """
        返回数据集中的组数
        """
        return len(self.images_info)

test_dataset = group_pred_dataset(images_info=images_info, transform=test_transform)
test_dl = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, shuffle=False, pin_memory=args.num_workers>1)

#### Datasets & DataLoaders
groups = [TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, M=args.M, N=args.N,
                       min_images_per_class=args.min_images_per_class
                       ) for n in range(args.N * args.N)]

if 'efficientnet' in args.backbone.lower():
    backbone_info = {
        # 'input_size': (210, 280),
        'input_size': args.train_resize,
        'layers_to_freeze': 5
    }

if 'mixvpr' in args.aggregator.lower():
    agg_config = {
        # 'in_channels' : 1280,
        # 'in_h' : 12,
        # 'in_w' : 15,
        'out_channels' : 640,
        'mix_depth' : 4,
        'mlp_ratio' : 1,
        'out_rows' : 4,
    } # the output dim will be (out_rows * out_channels)

model = GeoClassNet(args.backbone, backbone_info=backbone_info,aggregator=args.aggregator,agg_config=agg_config).to(args.device)

commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

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

class_subset_count = 2

# 在推理部分添加时间统计
inference_start_time = time.time()
lr_str, gpu_mem_list, cpu_mem_list = he_test.inference_time_count(args, model, classifiers, test_dl, groups, images_info, class_subset_count)
total_inference_time = time.time() - inference_start_time
avg_inference_time = total_inference_time / len(images_info)
avg_gpu_mem = np.mean(gpu_mem_list)
avg_cpu_mem = np.mean(cpu_mem_list)

print_str = f'检索前{class_subset_count}类\n' + \
            f'Average cut time per image: {avg_cut_time:.4f}s\n' + \
            f'Average inference time per image: {avg_inference_time:.4f}s\n' + \
            lr_str + '\n' + \
            f'Crop:\nAverage CPU memory usage: {avg_cpu_crop_mem:.4f}MB\n' + \
            f'VPR:\nAverage GPU memory usage: {avg_gpu_mem:.4f}MB\nAverage CPU memory usage: {avg_cpu_mem:.4f}MB'

            
# 打印统计信息
logging.info(print_str)