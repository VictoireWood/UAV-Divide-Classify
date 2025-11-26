import warnings

# 忽略UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as T
import logging
from datetime import datetime
import sys
import torchmetrics
from tqdm import tqdm
from math import sqrt
import numpy as np
import platform

from dataloaders.HCDataset import realHCDataset_N, InfiniteDataLoader, HCDataset_shN, TestDataset
from models import helper, regression
import commons

from utils.checkpoint import save_checkpoint_with_groups, resume_model_with_classifiers, resume_train_with_groups, resume_train_with_groups_all, save_checkpoint_with_groups_best_val
from utils.inference import inference_with_groups, inference_with_groups_with_val
from utils.losses import CoLoss
from utils.utils import move_to_device
from models.classifiers import AAMC
import parser

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
args = parser.parse_arguments()

assert args.train_set_path is not None, 'you must specify the train set path'
assert os.path.exists(args.train_set_path), 'train set path must exist'
# assert args.val_set_path is not None, 'you must specify the val set path'
assert (args.test_set_path is not None) or (args.val_set_path is not None), 'you must specify the test set path'
if args.test_set_path is not None:
    assert os.path.exists(args.test_set_path), 'test set path must exist'
if args.val_set_path is not None:
    assert os.path.exists(args.val_set_path), 'val set path must exist (real photo)'



# Parser变量
foldernames=['2013', '2017', '2019', '2020', '2022', '0001', '0002', '0003', '0004', 'real_photo', 'ct01', 'ct02']
# train_dataset_folders = ['2013', '2017', '2019', '2020', '2022', '0001', '0002', '0003', '0004']
# train_dataset_folders = ['2022']
# train_dataset_folders = ['ct01']
train_dataset_folders = ['ct02']
# test_datasets = ['real_photo', '2022', '2020']
# test_datasets = ['2022', '2020']
test_datasets = train_dataset_folders
test_datasets = ['real_photo']

if args.dataset_name == 'ct01':
    train_dataset_folders = ['ct01']
    test_datasets = ['ct01']
elif args.dataset_name == 'ct02':
    train_dataset_folders = ['ct02']
    test_datasets = ['ct02']
elif args.dataset_name == '2022':
    train_dataset_folders = ['2022']
    test_datasets = ['real_photo']

test_dataset = args.test_set_list

if 'dinov2' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'num_trainable_blocks': args.num_trainable_blocks,
    }
elif 'efficientnet_v2' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': args.layers_to_freeze,
    }
elif 'efficientnet' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': args.layers_to_freeze,
    }
elif 'resnet' in args.backbone.lower():
    backbone_info = {
        'input_size': args.train_resize,
        'layers_to_freeze': args.layers_to_freeze,
        'layers_to_crop': list(args.layers_to_crop),
    }
agg_config = {}


train_transform = T.Compose([
    T.Resize(args.train_resize, antialias=True),
    T.ColorJitter(brightness=0.2, contrast=0.3, saturation=0.3, hue=0.2),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform =T.Compose([
    T.Resize(args.test_resize, antialias=True),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

#### 初始化
commons.make_deterministic(args.seed)
commons.setup_logging(args.save_dir, console="info")
logging.info(" ".join(sys.argv))
logging.info(f"Arguments: {args}")
logging.info(f"The outputs are being saved in {args.save_dir}")

logging.info(f"train_dataset_folders: {train_dataset_folders}")
logging.info(f"test_datasets_folders: {test_datasets}")

#### Dataset & Dataloader
# if args.train_photos:

groups = []
for n in range(args.N):
    group = HCDataset_shN(group_num=n, dataset_name=args.dataset_name,train_path=args.train_set_path, train_dataset_folders=train_dataset_folders, M=args.M, N=args.N,min_images_per_class=args.min_images_per_class,transform=train_transform)
    groups.append(group)

# train_dataset = HCDataset(train_dataset_folders, random_sample_from_each_place=True, transform=train_transform, base_path=args.train_set_path)
# train_dataloader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)
# train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)

# ANCHOR 把一个分类器增加到N个分类器



# train_dl = DataLoader(train_dataset, batch_size=, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False, persistent_workers=args.num_workers>0)
# iterations_num = len(train_dataset) // train_batch_size
# logging.info(f'Found {len(train_dataset)} images in the training set.' )


test_dataset_list = []
test_datasets_load = test_datasets
if 'real_photo' in test_datasets:
    real_photo_dataset = realHCDataset_N(base_path=args.val_set_path, M=args.M, N=args.N, transform=test_transform)
    test_datasets_load.remove('real_photo')
    test_dataset_list.append(real_photo_dataset)
if len(test_datasets_load) != 0:
    fake_photo_dataset = TestDataset(test_folder=args.test_set_path, test_datasets=test_datasets, M=args.M, N=args.N, image_size=args.test_resize)
    test_dataset_list.append(fake_photo_dataset)
if len(test_dataset_list) > 1:
    test_dataset = ConcatDataset(test_dataset_list)
else:
    test_dataset = test_dataset_list[0]

test_img_num = len(test_dataset)
logging.info(f'Found {test_img_num} images in the test set.' )
test_num_workers = 2 if (args.device == "cuda" and platform.system() == "Linux") else 0
test_dl = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=test_num_workers, pin_memory=(args.device == "cuda"))

#### model
# model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config, regression_ratio=args.regression_ratio)
model = helper.HeightFeatureNet(backbone_arch=args.backbone, backbone_info=backbone_info, agg_arch=args.aggregator, agg_config=agg_config)
# classifier = AAMC(in_features=model.feature_dim, out_features=classes_num, s=args.aamc_s, m=args.aamc_m)
# NOTE 分类器输出为类的数量的2倍，这里类的数量为12。

model = model.to(args.device)

# TODO 多个分类器进行训练

classifiers = [AAMC(in_features=model.feature_dim, out_features=group.get_classes_num(), s=args.aamc_s, m=args.aamc_m) for group in groups]

classifiers_optimizers = [torch.optim.Adam(classifier.parameters(), lr=args.classifier_lr) for classifier in classifiers]

logging.info(f"Using {len(groups)} groups")
logging.info(f"The {len(groups)} groups have respectively the following number of classes {[g.get_classes_num() for g in groups]}")
logging.info(f"The {len(groups)} groups have respectively the following number of images {[g.get_images_num() for g in groups]}")
logging.info(f"Feature dim: {model.feature_dim}")
logging.info(f"resume_model: {args.resume_model}")


# 看model的可训练参数多少
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logging.info(f'Trainable parameters: {params/1e6:.4}M')

#### OPTIMIZER & SCHEDULER
# classifier_optimizer = optim.Adam(classifier.parameters(), lr=args.classifier_lr)
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, verbose=True) #NOTE: 学习率变化设置

#### Resume
if args.resume_model is not None:
    # model, classifier = resume_model(model, classifier)
    model, classifiers = resume_model_with_classifiers(model, classifiers)
elif args.resume_train is not None:
    # model, optimizer, best_loss, start_epoch_num = resume_train_with_params(model, optimizer, scheduler)

    # model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num = \
    #     resume_train_with_groups(args.save_dir, model, optimizer, classifiers, classifiers_optimizers)
    
    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num, scheduler, best_val_lr = \
        resume_train_with_groups_all(args.save_dir, model, optimizer, classifiers, classifiers_optimizers, scheduler)
    
    epoch_num = start_epoch_num - 1
    best_loss = best_train_loss
    best_val = best_val_lr
    logging.info(f"Resuming from epoch {start_epoch_num} with best train loss {best_train_loss:.2f} " +
                 f"from checkpoint {args.resume_train}")
else:
    best_valid_acc = 0
    start_epoch_num = 0
    best_loss = float('inf')
    best_val = 0.0


### Train&Loss
# 初始化模型、损失函数和优化器
cross_entropy_loss = torch.nn.CrossEntropyLoss()    #NOTE: 交叉熵损失，应该是softmax通用的loss形式


# 训练模型
scaler = torch.GradScaler('cuda')
for epoch_num in range(start_epoch_num, args.epochs_num):
    if optimizer.param_groups[0]['lr'] < 1e-6:
        logging.info('LR dropped below 1e-6, stopping training...')
        break

    classes_num = 0
    for g in groups:
        classes_num += g.get_classes_num()
    classes_num_list = [g.get_classes_num() for g in groups]

    train_loss = torchmetrics.MeanMetric().to(args.device)

    # Select classifier and dataloader according to epoch

    current_group_num = epoch_num % len(classifiers)
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=classes_num_list[current_group_num]).to(args.device) # EDIT 3 train_acc
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    move_to_device(classifiers_optimizers[current_group_num], args.device)



    #### Train
    # train_dataloader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=(args.device == "cuda"), drop_last=False)
    train_dataloader = InfiniteDataLoader(groups[current_group_num], 
                                          num_workers=args.num_workers,
                                          batch_size=args.batch_size, shuffle=True,
                                          pin_memory=(args.device == "cuda"), drop_last=True)
    dataloader_iterator = iter(train_dataloader)
    model = model.train()

    tqdm_bar = tqdm(range(args.iterations_per_epoch), ncols=100, desc="")
    #NOTE: tqmd.tqmd修饰一个可迭代对象，返回一个与原始可迭代对象完全相同的迭代器，但每次请求值时都会打印一个动态更新的进度条。
    for iteration in tqdm_bar:
        images, labels, _ = next(dataloader_iterator)   # NOTE return tensor_image, class编号, class_center_current
        images, labels = images.to(args.device), labels.to(args.device)

        optimizer.zero_grad()
        classifiers_optimizers[current_group_num].zero_grad()

        # with torch.cuda.amp.autocast('cuda',enabled=True): # NOTE: 加上了'cuda'参数和enabled  
        # ORIGION 
        with torch.autocast('cuda'):    # EDIT
            # LINK: https://pytorch.org/docs/stable/amp.html
            descriptors = model(images)
            # 1) 'output' is respectively the angular or cosine margin, of the AMCC or LMCC.
            # 2) 'logits' are the logits obtained multiplying the embedding for the
            # AMCC/LMCC weights. They are used to compute tha accuracy on the train batches
            output, logits = classifiers[current_group_num](descriptors, labels)
            loss = cross_entropy_loss(output, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.step(classifiers_optimizers[current_group_num])
        scaler.update()

        train_acc.update(logits, labels)    # ORIGION
        train_loss.update(loss.item())
        tqdm_bar.set_description(f"{loss.item():.1f}")
        del loss, images, output
        _ = tqdm_bar.refresh()  # ORIGION
        _ = tqdm_bar.update()   # EDIT

    # classifier = classifier.cpu()
    # move_to_device(classifier_optimizer, 'cpu') 

    #### Validation
    # correct_class_recall, threshold_recall = inference_with_groups(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)

    correct_class_recall, threshold_recall, val_lr = inference_with_groups_with_val(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)

    train_acc = train_acc.compute() * 100  
    train_loss = train_loss.compute()

    if train_loss < best_loss:
        is_best = True
        best_loss = train_loss
    else:
        is_best = False

    if val_lr > best_val:
        is_best_val = True
        best_val = val_lr
    else:
        is_best_val = False
        

    logging.info(f"E{epoch_num: 3d}, train_acc: {train_acc.item():.1f}, " +
                 f"train_loss: {train_loss.item():.2f}, best_train_loss: {scheduler.best:.2f}, " +
                 f"not improved for {scheduler.num_bad_epochs}/{args.scheduler_patience} epochs, " +
                 f"lr: {round(optimizer.param_groups[0]['lr'], 21)}, " +
                 f"classifier_lr: {round(classifiers_optimizers[current_group_num].param_groups[0]['lr'], 21)}")
    logging.info(f"E{epoch_num: 3d}, {correct_class_recall}")
    logging.info(f"E{epoch_num: 3d}, {threshold_recall}")

    scheduler.step(train_loss)

    # save_checkpoint_with_groups({"epoch_num": epoch_num + 1,
    #     "model_state_dict": model.state_dict(),
    #     "optimizer_state_dict": optimizer.state_dict(),
    #     "classifiers_state_dict": [c.state_dict() for c in classifiers],
    #     "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
    #     "args": args,
    #     "best_train_loss": best_loss,
    # }, is_best, args.save_dir)

    save_checkpoint_with_groups_best_val({"epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "args": args,
        "best_train_loss": best_loss,
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_lr": best_val,
    }, is_best, is_best_val, args.save_dir)


    torch.cuda.empty_cache()

correct_class_recall, threshold_recall = inference_with_groups(args=args, model=model, classifiers=classifiers, test_dl=test_dl, groups=groups, num_test_images=test_img_num)

logging.info(f"Test LR: {correct_class_recall}, {threshold_recall}")
print("Training complete.")