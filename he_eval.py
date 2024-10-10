
import sys
import torch
import logging

import he_test
import util
import parser
import commons
from classifiers import AAMC, LMCC, LinearLayer
from he_datasets import initialize, TrainDataset, TestDataset
from models.helper import GeoClassNet

args = parser.parse_arguments()
# assert args.train_set_path is not None, 'you must specify the train set path'
assert args.test_set_path is not None, 'you must specify the test set path'

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
        'out_channels' : 640,
        'mix_depth' : 4,
        'mlp_ratio' : 1,
        'out_rows' : 4,
    } # the output dim will be (out_rows * out_channels)
elif 'gem' in args.aggregator.lower():
    agg_config={
        'p': 3,
    }

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

test_dataset = TestDataset(args.test_set_path, M=args.M, N=args.N, image_size=args.test_resize)
test_dl = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

#### Model

# Each group has its own classifier, which depends on the number of classes in the group
if args.classifier_type == "AAMC":
    classifiers = [AAMC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "LMCC":
    classifiers = [LMCC(model.feature_dim, group.get_classes_num(), s=args.lmcc_s, m=args.lmcc_m) for group in groups]
elif args.classifier_type == "FC_CE":
    classifiers = [LinearLayer(model.feature_dim, group.get_classes_num()) for group in groups]

logging.info(f"Feature dim: {model.feature_dim}")

if args.resume_model is not None:
    model, classifiers = util.resume_model(args, model, classifiers)
else:
    logging.info("WARNING: You didn't provide a path to resume the model (--resume_model parameter). " +
                 "Evaluation will be computed using randomly initialized weights.")

lr_str = he_test.inference(args, model, classifiers, test_dl, groups, len(test_dataset))
logging.info(f"LR: {lr_str}")
