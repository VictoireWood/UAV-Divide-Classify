
import torch
from torch import nn
import torch.nn.functional as F
from models.backbones import vision_transformer as vit
import models.aggregators as aggregators
import models.backbones as backbones


def get_dinov2_backbone(arch:str, pretrained, backbone_info): # NOTE: 这是获得MulConv模块的入口！
    foundation_model_path = backbone_info['foundation_model_path']
    # img_size = backbone_info['input_size']
    img_size = 518
    # img_size = (img_size[0] // 14 * 14, img_size[1] // 14 * 14)
    if 'vits' in arch.lower():
        backbone = vit.vit_small(patch_size=14,img_size=img_size,init_values=1,block_chunks=0)
    elif 'vitb' in arch.lower():
        backbone = vit.vit_base(patch_size=14,img_size=img_size,init_values=1,block_chunks=0)
    elif 'vitl' in arch.lower():
        backbone = vit.vit_large(patch_size=14,img_size=img_size,init_values=1,block_chunks=0)
    elif 'vitg' in arch.lower():
        backbone = vit.vit_giant2(patch_size=14,img_size=img_size,init_values=1,block_chunks=0)

    if pretrained:
        assert foundation_model_path is not None, "Please specify foundation model path."
        model_dict = backbone.state_dict()
        state_dict = torch.load(foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)
    return backbone


def get_backbone(backbone_arch:str='dinov2_vitb14', pretrained:bool=True, backbone_info:dict={}):
    if 'dinov2' in backbone_arch.lower():
        if backbone_info['scheme'] == 'adapter':
            backbone = get_dinov2_backbone(arch=backbone_arch, pretrained=pretrained, backbone_info=backbone_info)
        elif backbone_info['scheme'] == 'finetune':
            backbone = backbones.DINOv2(model_name=backbone_arch, num_trainable_blocks=backbone_info['num_trainable_blocks'])
        return backbone
    elif 'efficientnet_v2' in backbone_arch.lower():
        layers_to_freeze = backbone_info['layers_to_freeze']
        layers_to_crop = backbone_info['layers_to_crop']
        return backbones.EfficientNet_V2(backbone_arch, pretrained, layers_to_freeze, layers_to_crop)
    


def get_aggregator(agg_arch:str='MixVPR', agg_config:dict={}):
    """Helper function that returns the aggregation layer given its name.
    If you happen to make your own aggregator, you might need to add a call
    to this helper function.

    Args:
        agg_arch (str, optional): the name of the aggregator. Defaults to 'ConvAP'.
        agg_config (dict, optional): this must contain all the arguments needed to instantiate the aggregator class. Defaults to {}.

    Returns:
        nn.Module: the aggregation layer
    """
    
    if 'cosplace' in agg_arch.lower():
        agg_config_tmp = {key:value for key,value in agg_config.items() if key == 'in_dim' or key == 'out_dim'}
        assert 'in_dim' in agg_config_tmp
        assert 'out_dim' in agg_config_tmp
        return aggregators.CosPlace(**agg_config_tmp)

    elif 'gem' in agg_arch.lower():
        agg_config_tmp = {key:value for key,value in agg_config.items() if key == 'p' or key == 'eps'}
        if agg_config_tmp == {}:
            agg_config_tmp['p'] = 3
            agg_config['p'] = 3
        else:
            assert 'p' in agg_config_tmp
        return aggregators.GeMPool(**agg_config_tmp)
    
    elif 'convap' in agg_arch.lower():
        agg_config_tmp = {key:value for key,value in agg_config.items() if key == 'in_channels'}
        assert 'in_channels' in agg_config_tmp
        return aggregators.ConvAP(**agg_config_tmp)
    
    elif 'mixvpr' in agg_arch.lower():
        agg_config_tmp = {key:value for key,value in agg_config.items() if key == 'in_channels' or key == 'out_channels' or key == 'in_h' or key == 'in_w' or key == 'mix_depth'}
        assert 'in_channels' in agg_config_tmp
        assert 'out_channels' in agg_config_tmp
        assert 'in_h' in agg_config_tmp
        assert 'in_w' in agg_config_tmp
        assert 'mix_depth' in agg_config_tmp
        return aggregators.MixVPR(**agg_config_tmp)
    elif 'avgpool' in agg_arch.lower():
        agg = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),   #REVIEW: 自适应池化
            Flatten(),
        )
        return agg

def freeze_dinov2_train_adapter(model:nn.Module):
    ## Freeze parameters except adapter
    # for name, param in model.module.backbone.named_parameters():  # ANCHOR 原始
    for name, param in model.backbone.named_parameters():
        if "adapter" not in name:
            param.requires_grad = False
    return model

def init_adapter(model:nn.Module):
## initialize Adapter
    for n, m in model.named_modules():
        if 'adapter' in n:
            for n2, m2 in m.named_modules():
                if 'D_fc2' in n2:
                    if isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0.)
                        nn.init.constant_(m2.bias, 0.)
            for n2, m2 in m.named_modules():
                if 'conv' in n2:
                    if isinstance(m2, nn.Conv2d):
                        nn.init.constant_(m2.weight, 0.00001)
                        nn.init.constant_(m2.bias, 0.00001)
    return model

class GeoClassNet(nn.Module):
    def __init__(self, backbone, backbone_info, aggregator, agg_config):
        super().__init__()
        # self.backbone, out_channels = get_backbone(backbone, pretrained=True,backbone_info=backbone_info)   # FIXME: 这里的out_channels没有
        self.backbone = get_backbone(backbone, pretrained=True,backbone_info=backbone_info)
        DINOV2_ARCHS = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }
        if 'dinov2' in backbone.lower():
            agg_config['in_channels'] = DINOV2_ARCHS[backbone]
        input_size = backbone_info['input_size']
        # bb_out_dim = get_output_dim(self.backbone, (32, 3, input_size[0], input_size[1]))
        # self.pool = get_pooling()
        agg_config['in_h'] = input_size[0] // 14
        agg_config['in_w'] = input_size[1] // 14
        if aggregator == 'MixVPR':
            agg_config['out_channels'] = agg_config['in_channels'] // 2 # REVIEW
            pass
            self.aggregator = get_aggregator(agg_arch=aggregator, agg_config=agg_config)
            out_channels = agg_config['out_channels'] * agg_config['out_rows']   # EDIT
        else:
            self.aggregator = get_aggregator(agg_arch=aggregator, agg_config=agg_config)
            out_channels = self.aggregator(torch.zeros(1, agg_config['in_channels'], agg_config['in_h'], agg_config['in_w'])).shape[1]
        self.agg_config = agg_config
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            L2Norm()
        )
        self.feature_dim = out_channels
        self.backbone_arch = backbone
        self.backbone_info = backbone_info

    def forward(self, x):
        if 'dinov2' in self.backbone_arch:
            if self.backbone_info['scheme'] == 'adapter':
                x = self.backbone(x)['x_norm_patchtokens']
                x = x.view(x.shape[0], self.agg_config['in_h'], self.agg_config['in_w'], x.shape[2]).permute(0, 3, 1, 2)
            else:
                x = self.backbone(x)
        else:
            x = self.backbone(x)
        # x = self.pool(x)          # ANCHOR
        x = self.aggregator(x)
        # x = self.classifier(x)    # ORIGIN
        return x

def get_output_dim(model, input_size=(32, 3, 210, 280)):
    """Return the number of channels in the output of a model."""
    return model(torch.ones(input_size)).shape[1:]

def get_pooling():
    pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),   #REVIEW: 这里是GeM吗？自适应池化
            Flatten(),
        )
    return pooling

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        """
        #NOTE: 
        Forward pass of the Flatten module.
        forward: This is the forward pass method. It checks if the input tensor x has a shape where the second and third dimensions (height and width) are equal and equal to 1. If the assertion fails, it raises an error. Otherwise, it returns the flattened version of the tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Flattened tensor of shape (batch_size, channels).

        Raises:
            AssertionError: If the height and width dimensions of the input tensor are not equal to 1.
        """
        assert x.shape[2] == x.shape[3] == 1, f"{x.shape[2]} != {x.shape[3]} != 1"
        return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

