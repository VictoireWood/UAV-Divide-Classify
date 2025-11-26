
import sys
import torch
import logging
import torchmetrics
from tqdm import tqdm
import torchvision.transforms as T
from torch import optim
from torch.utils.data import DataLoader

import he_test
import util
from models import helper
import parser
import commons
from he_datasets import TrainDataset, TestDataset  # ANCHOR
# from datasets_M import TrainDataset, TestDataset    # EDIT
import numpy as np

# 有高度估计输入的定高模型(150)

from classifiers import AAMC, LMCC, LinearLayer, ACLC, QAMC

args = parser.parse_arguments()
assert args.train_set_path is not None, 'you must specify the train set path'
# assert args.val_set_path is not None, 'you must specify the val set path'   # NOTE: 其实val这个文件夹根本没有用到
assert args.test_set_path is not None, 'you must specify the test set path'

commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

#### Datasets & DataLoaders
# ORIGION
# train_augmentation = T.Compose([
#         T.ToTensor(),
#         T.Resize(args.train_resize),
#         T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1]),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
# EDIT 李春雨
# train_augmentation = T.Compose([
#         T.Resize(args.train_resize, antialias=True),
#         T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
#         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
#         T.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=15),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])
# EDIT 邵星雨
# train_augmentation = T.Compose([
#         T.Resize(args.train_resize, antialias=True),
#         T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1+0.34], antialias=True),
#         T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
#         T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
#         T.ToTensor(),
#         T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ])

train_augmentation = T.Compose([
        T.Resize(args.train_resize, antialias=True),
        T.RandomResizedCrop((args.train_resize[0], args.train_resize[1]), scale=(1-0.34, 1), ratio=(1.25, 1.4), antialias=True),
        # T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
        # T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

groups = [TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, 
                       M=args.M, N=args.N,
                       min_images_per_class=args.min_images_per_class,
                       transform=train_augmentation
                       ) for n in range(args.N * args.N)]
# SECTION
# groups = []
# for n in range(args.N * args.N):
#     '''# ORIGION 没用到自适应M
#     group = TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, M=args.M, N=args.N,
#                        min_images_per_class=args.min_images_per_class,
#                        transform=train_augmentation
#                        )'''
#     # EDIT 自适应M，不需要另外输入
#     group = TrainDataset(args.train_set_path, dataset_name=args.dataset_name, group_num=n, N=args.N,
#                        min_images_per_class=args.min_images_per_class,
#                        transform=train_augmentation
#                        )
#     groups.append(group)
# !SECTION

#NOTE: 对应论文里的group，一个group对应一个classifier，也就是同一个颜色的方块集合，一共有N*N组，文章里对应2*2=4组

# val_dataset = TestDataset(args.val_set_path, M=args.M, N=args.N, image_size=args.test_resize)   # ORIGION: val_dataset其实没用
test_dataset = TestDataset(args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize)     # EDIT v1：还没有用到自适应M
# test_dataset = TestDataset(args.test_set_path, N=args.N, image_size=args.test_resize)     # EDIT v2：用到自适应M

# val_dl = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)  # ORIGION: val_dl最后没用
test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

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
        'layers_to_freeze': 8
    }
elif 'efficientnet' in args.backbone.lower():
    backbone_info = {
        # 'input_size': (210, 280),
        'input_size': args.train_resize,
        'layers_to_freeze': 3
    }
elif 'radio' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'num_trainable_blocks': args.train_blocks_num,
        'return_token': args.return_token,
        'pre_norm': args.pre_norm,
    }

if args.aggregator == None:
    agg_config = {}
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
else:
    agg_config = {}

# agg_config={
#     # 'in_channels' : 1280,
#     # 'in_h' : 12,
#     # 'in_w' : 15,
#     'out_channels' : 640,
#     'mix_depth' : 4,
#     'mlp_ratio' : 1,
#     'out_rows' : 4,
# }   # the output dim will be (out_rows * out_channels)


## GeM
# agg_config={
#     'p': 3,
# }

model = helper.GeoClassNet(args.backbone, backbone_info=backbone_info,aggregator=args.aggregator,agg_config=agg_config)

model = model.to(args.device)

if 'dinov2' in args.backbone.lower() and backbone_info['scheme']=='adapter':
    model = helper.freeze_dinov2_train_adapter(model)
    model = helper.init_adapter(model)

# 看model的可训练参数多少
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logging.info(f'Trainable parameters: {params/1e6:.4}M')

# model = torch.nn.DataParallel(model)  # REVIEW 这里是CricaVPR的并行计算，这里去掉

# NEW: Adaptive Curriculum Learning Loss 我改成ACLC(Adaptive Curriculum Learning Classifier)

# Each group has its own classifier, which depends on the number of classes in the group
if args.classifier_type == "AAMC":
    classifiers = [AAMC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "LMCC":
    classifiers = [LMCC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
# NOTE: LMMC是CosPlace里用的CosFace Loss，AAMC是本文章中用的ArcFace Loss
elif args.classifier_type == "FC_CE":
    classifiers = [LinearLayer(model.feature_dim, group.get_classes_num()) for group in groups]
    #REVIEW: LinearLayer这是什么？
elif args.classifier_type == "ACLC":
    classifiers = [ACLC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "QAMC":
    classifiers = [QAMC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]


classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[g.get_classes_num() for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")
logging.info(f"Feature dim: {model.feature_dim}")
logging.info(f"resume_model: {args.resume_model}")

if args.resume_model is not None:   # FIXME: 这里需要根据cricaVPR改一下
    model, classifiers = util.resume_model(args, model, classifiers)

cross_entropy_loss = torch.nn.CrossEntropyLoss()    #NOTE: 交叉熵损失，应该是softmax通用的loss形式

#### OPTIMIZER & SCHEDULER
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# ORIGION
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True) #NOTE: 学习率变化
# EDIT
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience)

#### Resume
if args.resume_train:   # FIXME: 这里需要根据cricaVPR改一下
    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num, scheduler = \
        util.resume_train_with_groups(args, args.save_dir, model, optimizer, classifiers, classifiers_optimizers, scheduler)
    epoch_num = start_epoch_num - 1
    best_loss = best_train_loss
    logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_valid_acc = 0
    start_epoch_num = 0
    best_loss = 100

# scaler = torch.cuda.amp.GradScaler('cuda')  # NOTE: 加上了'cuda'参数
# ORIGION

test_lr_str = he_test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))  # ANCHOR
# test_lr_str, test_h_str = test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))  # REVIEW

logging.info(f"Test LR: {test_lr_str}")

# logging.info(f"Test height LR: {test_h_str}")   # EDIT 加上高度分类正确率的百分比
