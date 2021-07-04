import torch,math
import torch.nn as nn

from . import resnetbackbone
from . import densenetbackbone

class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    came from : https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/model/utils/gelu.py
    """
    def __init__(self):

        super(GELU,self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class Conv2dBA(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        bias = False,
        bn = False,
        act = False
    ):
        super(Conv2dBA,self).__init__()

        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size,stride = stride,padding = padding,bias = bias)
        self.bn = nn.BatchNorm2d(out_channels,eps = 1e-8) if bn else None
        self.act = nn.ReLU() if act else None
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x) if (self.bn is not None) else x
        x = self.act(x) if (self.act is not None) else x
        return x

class Extactor(nn.Module):

    def __init__(
        self,
        bb_name,
        backbone_pretrained_path = None
    ):
            
        super(Extactor,self).__init__()

        # region feature extraction backbone
        assert bb_name in ["resnet50","densenet161"]
        if bb_name == "resnet50":
            self.backbone = resnetbackbone.resnet50()
            if backbone_pretrained_path is not None:
                try:
                    self.backbone.load_state_dict(torch.load(backbone_pretrained_path))
                    print("backbone model weights loading finished!")
                except Exception as e:
                    print("backbone model weights loading error!",e)
            del self.backbone.fc
            del self.backbone.avgpool
        elif bb_name == "densenet161":
            if backbone_pretrained_path is not None:
                self.backbone = densenetbackbone.densenet161(True,backbone_pretrained_path)
            else:
                self.backbone = densenetbackbone.densenet161()
            del self.backbone.features.norm5
            del self.backbone.classifier
        else:
            raise ValueError
        
        # endregion

    def forward(self,img):
        return self.backbone(img)