
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

# 有高度估计输入的定高模型(150)

from classifiers import AAMC, LMCC, LinearLayer, ACLC

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
train_augmentation = T.Compose([
        T.Resize(args.train_resize, antialias=True),
        T.RandomResizedCrop([args.train_resize[0], args.train_resize[1]], scale=[1-0.34, 1], antialias=True),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),  
        T.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=15),
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
elif 'gem' in args.aggregator.lower():
    agg_config={
        'p': 3,
    }

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

if 'dinov2' in args.backwbone.lower() and backbone_info['scheme']=='adapter':
    model = helper.freeze_dinov2_train_adapter(model)
    model = helper.init_adapter(model)

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
    model, model_optimizer, classifiers, classifiers_optimizers, best_train_loss, start_epoch_num = \
        util.resume_train_with_groups(args, args.save_dir, model, optimizer, classifiers, classifiers_optimizers)
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
scaler = torch.GradScaler('cuda')  # NOTE: 加上了'cuda'参数
# LINK: https://pytorch.org/docs/stable/amp.html
for epoch_num in range(start_epoch_num, args.epochs_num):
    if optimizer.param_groups[0]['lr'] < 1e-6:
        logging.info('LR dropped below 1e-6, stopping training...')
        break
    # train_acc = torchmetrics.Accuracy().to(args.device) # ORIGION train_acc
    # SECTION
    classes_num = 0
    for g in groups:
        classes_num += g.get_classes_num()
    classes_num_list = [g.get_classes_num() for g in groups]
    # !SECTION
    # train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=classes_num).to(args.device) # EDIT 1 train_acc
    # train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=classes_num_list[]).to(args.device) # EDIT 2 train_acc
    train_loss = torchmetrics.MeanMetric().to(args.device)

    # Select classifier and dataloader according to epoch
    current_group_num = epoch_num % len(classifiers)
    train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=classes_num_list[current_group_num]).to(args.device) # EDIT 3 train_acc
    classifiers[current_group_num] = classifiers[current_group_num].to(args.device)
    util.move_to_device(classifiers_optimizers[current_group_num], args.device)

    dataloader = commons.InfiniteDataLoader(groups[current_group_num], num_workers=args.num_workers,
                                            batch_size=args.batch_size, shuffle=True,
                                            pin_memory=(args.device == "cuda"), drop_last=True)
    #### Train
    dataloader_iterator = iter(dataloader)
    model = model.train()

    tqdm_bar = tqdm(range(args.iterations_per_epoch), ncols=100, desc="")
    #NOTE: tqmd.tqmd修饰一个可迭代对象，返回一个与原始可迭代对象完全相同的迭代器，但每次请求值时都会打印一个动态更新的进度条。
    for iteration in tqdm_bar:
        images, labels, _ = next(dataloader_iterator)   # return tensor_image, class_num, class_center
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
        # train_acc.update(logits.transpose(0,1),labels)  # EDIT
        train_loss.update(loss.item())
        tqdm_bar.set_description(f"{loss.item():.1f}")
        del loss, images, output
        _ = tqdm_bar.refresh()  # ORIGION
        _ = tqdm_bar.update()   # EDIT

    classifiers[current_group_num] = classifiers[current_group_num].cpu()
    util.move_to_device(classifiers_optimizers[current_group_num], "cpu")

    #### Validation
    val_lr_str = he_test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))

    train_acc = train_acc.compute() * 100
    train_loss = train_loss.compute()

    if train_loss < best_loss:
        is_best = True
        best_loss = train_loss
    else:
        is_best = False

    logging.info(f"E{epoch_num: 3d}, train_acc: {train_acc.item():.1f}, " +
                 f"train_loss: {train_loss.item():.2f}, best_train_loss: {scheduler.best:.2f}, " +
                 f"not improved for {scheduler.num_bad_epochs}/{args.scheduler_patience} epochs, " +
                 f"lr: {round(optimizer.param_groups[0]['lr'], 21)}, " +
                 f"classifier_lr: {round(classifiers_optimizers[current_group_num].param_groups[0]['lr'], 21)}")
    logging.info(f"E{epoch_num: 3d}, Val LR: {val_lr_str}") # NOTE 测试召回率？

    # logging.info(f"E{epoch_num: 3d}, Val height LR: {val_h_str}")   # EDIT 加上高度分类正确率的百分比

    scheduler.step(train_loss)
    util.save_checkpoint({"epoch_num": epoch_num + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "classifiers_state_dict": [c.state_dict() for c in classifiers],
        "optimizers_state_dict": [c.state_dict() for c in classifiers_optimizers],
        "args": args,
        "best_train_loss": best_loss
    }, is_best, args.save_dir)
    torch.cuda.empty_cache()

test_lr_str = he_test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))  # ANCHOR
# test_lr_str, test_h_str = test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))  # REVIEW

logging.info(f"Test LR: {test_lr_str}")

# logging.info(f"Test height LR: {test_h_str}")   # EDIT 加上高度分类正确率的百分比
