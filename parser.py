
import os
import argparse
from datetime import datetime

from generate_database import flight_heights    # EDIT

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # LMCC Parameters
    parser.add_argument("--lmcc_m", type=float, default=0.2,
                        help="margin m in LMCC layer")
    parser.add_argument("--lmcc_s", type=float, default=100.0,
                        help="param s in LMCC layer")
    parser.add_argument("-ct", "--classifier_type", type=str, default="AAMC",
                        choices=["AAMC", "LMCC", "FC_CE", "ACLC"],
                        help="Use the AAMC (arcFace), LMCC (cosFace), or FC_CE (fc + CE loss)")
                        # NOTE CE-Cross-Entropy Loss交叉熵损失   fc-Fully Connected 全连接层

    # Groups parameters
    parser.add_argument("--M", type=int, default=20,help="_")
    parser.add_argument("--N", type=int, default=2, help="_")
    parser.add_argument("--min_images_per_class", type=int, default=10,
                        help="minimum number of images per class. Otherwise class is discarded")
    # EDIT
    # parser.add_argument("--H", type=int, default=100)   # NOTE: 按照高度分组

    # Training parameters
    parser.add_argument("--seed", type=int, default=0, help="_")
    parser.add_argument("-ipe", "--iterations_per_epoch", type=int, default=2000, help="_")
    parser.add_argument("-bs", "--batch_size", type=int, default=64, help="_")
    parser.add_argument("--scheduler_patience", type=int, default=10, help="_")
    parser.add_argument("--epochs_num", type=int, default=500, help="_")
    # parser.add_argument("--train_resize", type=int, default=(224, 224), help="_") # ANCHOR
    # parser.add_argument("--train_resize", type=tuple, default=(360, 480), help="_") # REVIEW version 1
    # parser.add_argument("--train_resize", type=int, default=(222, 296), help="_")   # REVIEW    如果用DINOv2，就改成210*280
    # parser.add_argument("--train_resize", type=int, default=(210, 280), help="_")   # REVIEW    干脆DINOv2和CNN都用这个尺寸
    parser.add_argument("--train_resize", type=int, default=(336, 448), help="_")   # REVIEW    李老师用的是这个尺寸
    # parser.add_argument("--test_resize", type=int, default=256, help="_")           # ANCHOR
    # parser.add_argument("--test_resize", type=int, default=222, help="_")           # REVIEW
    # parser.add_argument("--test_resize", type=int, default=210, help="_")           # REVIEW   干脆DINOv2和CNN都用这个尺寸
    parser.add_argument("--test_resize", type=int, default=336, help="_")           # REVIEW   李老师用的是这个尺寸

    parser.add_argument("--lr", type=float, default=0.0001, help="_")
    parser.add_argument("--classifier_lr", type=float, default=0.01, help="_")
    parser.add_argument("-bb", "--backbone", type=str, default="EfficientNet_B0",
                        # choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7"],    # ANCHOR 原始
                        choices=["EfficientNet_B0", "EfficientNet_B5", "EfficientNet_B7", "EfficientNet_V2_M", "dinov2_vitb14", "dinov2_vits14"],  # REVIEW 邵星雨改
                        help="_")
    parser.add_argument("-ds","--dinov2_scheme", type=str, default="adapter", choices=["adapter", "finetune"], help="_")  # REVIEW 邵星雨加
    parser.add_argument("-tbn","--train_blocks_num", type=int, default=0, help="_")
    
    parser.add_argument("-agg", "--aggregator", type=str, default="MixVPR",
                        choices=["MixVPR", "CosPlace", "ConvAP", "GeMPool","AvgPool", "CricaVPR"],
                        help="_")   # EDIT
    
    parser.add_argument("-mcl","--model_classifier_layer", type=bool, default=True, help="_")

    # EDIT
    # Test parameters
    parser.add_argument('--threshold', type=int, default=None, help="验证是否成功召回的可容许偏差的距离，单位为米")    # REVIEW M自适应的话可以不设置


    # Init parameters
    parser.add_argument("--resume_train", type=str, default=None, help="path with *_ckpt.pth checkpoint")
    parser.add_argument("--resume_model", type=str, default=None, help="path with *_model.pth model")

    # Other parameters
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="_")
    parser.add_argument("--num_workers", type=int, default=16, help="_")

    # Paths parameters
    parser.add_argument("--exp_name", type=str, default="default",
                        help="name of experiment. The logs will be saved in a folder with such name")
    parser.add_argument("--dataset_name", type=str, default="sf_xl",
                        choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k","QingDao_Flight"], help="_")   # REVIEW
                        # choices=["sf_xl", "tokyo247", "pitts30k", "pitts250k"], help="_") # ANCHOR
    parser.add_argument("--train_set_path", type=str, default=None,
                        help="path to folder of train set")
    parser.add_argument("--val_set_path", type=str, default=None,
                        help="path to folder of val set")
    parser.add_argument("--test_set_path", type=str, default=None,
                        help="path to folder of test set")
    
    args = parser.parse_args()

    # EDIT
    if args.exp_name == "default":
        # args.exp_name = f'udc-{args.backbone}-{args.classifier_type}-{args.aggregator}-self.classifier_{args.model_classifier_layer}-{args.N}-{args.M}-h{flight_heights[0]}~{flight_heights[-1]}'
        if 'dinov2' in args.backbone.lower():
            args.exp_name = f'udc-{args.backbone}-{args.dinov2_scheme}-{args.classifier_type}-{args.aggregator}-self.classifier_{args.model_classifier_layer}-{args.N}-{args.M}-h{flight_heights[0]}~{flight_heights[-1]}'
            if args.dinov2_scheme == 'finetune':
                args.exp_name = f'{args.exp_name}-finetune_train_blocks_{args.train_blocks_num}'
        else:
            args.exp_name = f'udc-{args.backbone}-{args.classifier_type}-{args.aggregator}-self.classifier_{args.model_classifier_layer}-{args.N}-{args.M}-h{flight_heights[0]}~{flight_heights[-1]}'


    # if 'dinov2' in args.backbone.lower():
    #     args.train_resize = (210, 280)
    #     args.test_resize = 210

    args.save_dir = os.path.join("logs", args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    return args

