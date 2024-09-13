
import torch
import logging
import torchvision
from torch import nn
import torch.nn.functional as F


class GeoClassNet(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone, out_channels = get_backbone(backbone)
        self.pool = get_pooling()
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            L2Norm()
        )
        self.feature_dim = out_channels

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def get_pooling():
    pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),   #REVIEW: 这里是GeM吗？自适应池化
            Flatten(),
        )
    return pooling


def get_pretrained_torchvision_model(backbone_name):
    """This function takes the name of a backbone and returns the pretrained model from torchvision.
    Examples of backbone_name are 'ResNet18' or 'EfficientNet_B0'
    """
    try:  # Newer versions of pytorch require to pass weights=weights_module.DEFAULT
        weights_module = getattr(__import__('torchvision.models', fromlist=[f"{backbone_name}_Weights"]), f"{backbone_name}_Weights")
        model = getattr(torchvision.models, backbone_name.lower())(weights=weights_module.DEFAULT)
    except (ImportError, AttributeError):  # Older versions of pytorch require to pass pretrained=True
        model = getattr(torchvision.models, backbone_name.lower())(pretrained=True)
    return model


def get_backbone(backbone):
    model = get_pretrained_torchvision_model(backbone)

    for name, child in model.features.named_children():
        logging.debug("Freeze all EfficientNet layers up to n.5")
        if name == "5":
            break
        for params in child.parameters():
            params.requires_grad = False

    model = model.features
    out_channels = get_output_channels_dim(model)

    return model, out_channels


def get_output_channels_dim(model):
    """Return the number of channels in the output of a model."""
    return model(torch.ones([1, 3, 224, 224])).shape[1]


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

