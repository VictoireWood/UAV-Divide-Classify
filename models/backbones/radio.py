from PIL import Image

import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

import numpy as np

# RADIO_ARCHS = {
#     'radio_v2.5-h': 3840,
#     'radio_v2.1': 5120,
#     'radio_v2.1-l': ,
#     'dinov2_vitg16': 1536,
# }

# RADIO_ARCHS = ["radio_v2.5-h", "radio_v2.5-l", "radio_v2.5-b", "e-radio_v2"]

D = 1280


class RADIO(torch.nn.Module):
    """
    RADIO model

    Args:
        model_name (str): The name of the model architecture 
            should be one of ('dinov2_vits16', 'dinov2_vitb16', 'dinov2_vitl16', 'dinov2_vitg16')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """
    def __init__(
            self,
            model_name='radio_v2.5-h',
            num_trainable_blocks=1,
            pre_norm=False,
            return_token=False,
        ):
        super().__init__()  # NOTE - 调用nn.Module的初始化方法

        # assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        # self.model = torch.hub.load('NVlabs/RADIO', 'radio_model', version=model_name, progress=True, skip_validation=True)
        self.model = torch.hub.load('/root/.cache/torch/hub/NVlabs_RADIO_main', 'radio_model',version=model_name, progress=True, skip_validation=True, source='local')
        self.num_channels = D
        self.num_trainable_blocks = num_trainable_blocks
        self.pre_norm = pre_norm
        self.return_token = return_token

        if "e-radio" in model_name:
            self.model.model.set_optimal_window_size(x.shape[2:]) #where it expects a tuple of (height, width) of the input image.

        self.model.blocks[:-self.num_trainable_blocks].requires_grad_(False)
        for name, param in self.model.named_parameters():
            if 'patch_generator' in name:
                param.requires_grad_(False)

    

    def forward(self, x: torch.Tensor):
        """
        The forward method for the RADIO class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 16.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 16, W // 16].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        # model = self.model
        # x = Image.open('assets/radio_overview_github.png').convert('RGB')
        # x = pil_to_tensor(x).to(dtype=torch.float32, device='cuda')
        x = x.to(dtype=torch.float32, device='cuda')
        # x.div_(255.0)  # RADIO expects the input values to be between 0 and 1 
        # NOTE 这里由于之前的transform的ToTensor()已经将图像归一化了，这里不再需要再归一化一遍了
        # x = x.unsqueeze(0) # Add a batch dimension

        if x.shape[-2] % 16 != 0 or x.shape[-1] % 16 != 0:
            nearest_res = self.model.get_nearest_supported_resolution(*x.shape[-2:])
            # x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False)
            x = F.interpolate(x, nearest_res, mode='bilinear', align_corners=False, antialias=True)

        if self.pre_norm:
            conditioner = self.model.make_preprocessor_external()
            cond_x = conditioner(x)
            x = cond_x

        summary, spatial_features = self.model(x, feature_fmt='NCHW')
        assert spatial_features.ndim == 4

        if self.return_token:
            return spatial_features, summary
        return spatial_features


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params/1e6:.3}M')

if __name__ == '__main__':

    size = (360, 480)
    target_size = ((size[0]//16)*16, (size[1]//16)*16)
    cut_size_l = ((size[0]-target_size[0])//2, (size[1]-target_size[1])//2)
    cut_size_r = (size[0] - cut_size_l[0] - target_size[0], size[1] - cut_size_l[1] - target_size[1])
    size_r = (target_size[0] + cut_size_l[0], target_size[1] + cut_size_l[1])
    x = torch.randn(13, 3, size[0], size[1])
    x = x[:,:, cut_size_l[0]:size_r[0], cut_size_l[1]:size_r[1]]
    m = RADIO(model_name='dinov2_vitb16',
                        # pretrained=True,
                        # layers_to_freeze=7,
                        # layers_to_crop = [],
                        num_trainable_blocks = 2,
                        )
    r = m(x)
    print_nb_params(m)
    print(f'Input shape is {x.shape}')
    print(f'Output shape is {r.shape}')