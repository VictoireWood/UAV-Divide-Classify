import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6, work_with_tokens=False):
        super().__init__()
        self.p = Parameter(torch.ones(1)*p)
        self.eps = eps
        self.work_with_tokens=work_with_tokens
    def forward(self, x):
        return gem(x, p=self.p, eps=self.eps, work_with_tokens=self.work_with_tokens)
    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def gem(x, p=3, eps=1e-6, work_with_tokens=False):
    if work_with_tokens:
        x = x.permute(0, 2, 1)
        # unseqeeze to maintain compatibility with Flatten
        return F.avg_pool1d(x.clamp(min=eps).pow(p), (x.size(-1))).pow(1./p).unsqueeze(3)
    else:
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): assert x.shape[2] == x.shape[3] == 1; return x[:,:,0,0]

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        return F.normalize(x, p=2, dim=self.dim)

# 对一个数进行因式分解
def factorization(num):
    factor = []
    while num > 1:
        for i in range(num - 1):
            k = i + 2
            if num % k == 0:
                factor.append(k)
                num = int(num / k)
                break
    return factor


# class CricaVPRNet(nn.Module):
class CricaVPR(nn.Module):
    """The used networks are composed of a backbone and an aggregation layer.
    """
    def __init__(self, pretrained_foundation = False, foundation_model_path = None, in_channels = 768):
        super().__init__()
        # self.backbone = get_backbone(pretrained_foundation, foundation_model_path)
        self.aggregation = nn.Sequential(L2Norm(), GeM(work_with_tokens=None), Flatten())

        # In TransformerEncoderLayer, "batch_first=False" means the input tensors should be provided as (seq, batch, feature) to encode on the "seq" dimension.
        # Our input tensor is provided as (batch, seq, feature), which performs encoding on the "batch" dimension.
        
        # encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=16, dim_feedforward=2048, activation="gelu", dropout=0.1, batch_first=False)
        if in_channels == 768:
            nhead = 16
            dim_feedforward = 2048
        else:
            factors = factorization(in_channels)
            nhead = 1
            if len(factors) > 4:
                for f in factors[0:4]:
                    nhead = nhead * f
            else:
                for f in factors:
                    nhead = nhead * f
            if 3 in factors:
                dim_feedforward = in_channels * 8 // 3 
            else:
                dim_feedforward = in_channels * 4
            

        encoder_layer = nn.TransformerEncoderLayer(d_model=in_channels, nhead=nhead, dim_feedforward=dim_feedforward, activation="gelu", dropout=0.1, batch_first=False)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # Cross-image encoder
        # NOTE 两层transformer encoder的结构

    def forward(self, x):
        # x = self.backbone(x)        

        if isinstance(x, dict):
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            x0 = x["x_norm_clstoken"]
            x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0, 3, 1, 2) 


        elif isinstance(x, torch.Tensor):
            x_w, x_h = x.shape[-2, -1]
            x_p = x
            x0 = self.aggregation(x_p)

        layer_2_left_w = round(x_w / 2)   # 第二层金字塔左边的宽度
        layer_2_up_h = round(x_h / 2)
        # layer_2_right_w = x_w - layer_2_left_w
        # layer_2_down_h = x_h - layer_2_up_h


        layer_3_side_w = round(x_w / 3)   # 第三层金字塔两边的宽度
        layer_3_side_h = round(x_h / 3)
        layer_3_mid_w = x_w - layer_3_side_w * 2
        layer_3_mid_h = x_h - layer_3_side_h * 2

        # x10,x11,x12,x13 = self.aggregation(x_p[:,:,0:8,0:8]),self.aggregation(x_p[:,:,0:8,8:]),self.aggregation(x_p[:,:,8:,0:8]),self.aggregation(x_p[:,:,8:,8:])
        # x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.aggregation(x_p[:,:,0:5,0:5]),self.aggregation(x_p[:,:,0:5,5:11]),self.aggregation(x_p[:,:,0:5,11:]),\
        #                                 self.aggregation(x_p[:,:,5:11,0:5]),self.aggregation(x_p[:,:,5:11,5:11]),self.aggregation(x_p[:,:,5:11,11:]),\
        #                                 self.aggregation(x_p[:,:,11:,0:5]),self.aggregation(x_p[:,:,11:,5:11]),self.aggregation(x_p[:,:,11:,11:])
        # x = [i.unsqueeze(1) for i in [x0,x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]

        x10,x11,x12,x13 = self.aggregation(x_p[:,:,0:layer_2_up_h,0:layer_2_left_w]),\
                        self.aggregation(x_p[:,:,0:layer_2_up_h,layer_2_left_w:]),\
                        self.aggregation(x_p[:,:,layer_2_up_h:,0:layer_2_left_w]),\
                        self.aggregation(x_p[:,:,layer_2_up_h:,layer_2_left_w:])
        x20,x21,x22,x23,x24,x25,x26,x27,x28 = self.aggregation(x_p[:,:,0:layer_3_side_h,0:layer_3_side_w]),\
                                            self.aggregation(x_p[:,:,0:layer_3_side_h,layer_3_side_w:layer_3_side_w+layer_3_mid_w]),\
                                            self.aggregation(x_p[:,:,0:layer_3_side_h,layer_3_side_w+layer_3_mid_w:]),\
                                            self.aggregation(x_p[:,:,layer_3_side_h:layer_3_side_h+layer_3_mid_h,0:layer_3_side_w]),\
                                            self.aggregation(x_p[:,:,layer_3_side_h:layer_3_side_h+layer_3_mid_h,layer_3_side_w:layer_3_side_w+layer_3_mid_w]),\
                                            self.aggregation(x_p[:,:,layer_3_side_h:layer_3_side_h+layer_3_mid_h,layer_3_side_w+layer_3_mid_w:]),\
                                            self.aggregation(x_p[:,:,layer_3_side_h+layer_3_mid_h:,0:layer_3_side_w]),\
                                            self.aggregation(x_p[:,:,layer_3_side_h+layer_3_mid_h:,layer_3_side_w:layer_3_side_w+layer_3_mid_w]),\
                                            self.aggregation(x_p[:,:,layer_3_side_h+layer_3_mid_h:,layer_3_side_w+layer_3_mid_w:])
        x = [i.unsqueeze(1) for i in [x0,x10,x11,x12,x13,x20,x21,x22,x23,x24,x25,x26,x27,x28]]



        x = torch.cat(x,dim=1)
        # x = self.encoder(x).view(B,14*D)
        x = self.encoder(x).view(B,-1)
        x = torch.nn.functional.normalize(x, p=2, dim=-1)
        return x