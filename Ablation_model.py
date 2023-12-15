import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchstat import stat
import time
from mobilevit_v3_v2 import *
from ExternalAttentionWithChannelSimilarityMulAvg import ExternalAttention
from thop import profile
from torchstat import stat
from torchsummary import summary
#可学习的权重参数
class AdaptiveWeight(nn.Module):
    def __init__(self, init_value=1e-3):
        super().__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))
    def forward(self, input):
        return input * self.scale
#带有高通滤波的block
class Block(nn.Module):
    expansion=1
    def __init__(self,c1,c2,stride=1):
        super().__init__()
        #1*1
        self.conv1 = nn.Conv2d(c1,c1,kernel_size=3,padding=1)
        self.bn1= nn.BatchNorm2d(c1)
        self.down = nn.AvgPool2d(kernel_size=2)
        self.conv2=nn.Conv2d(2*c1,2*c1,kernel_size=3,padding=1,stride=stride,groups=2*c1)
        self.bn2= nn.BatchNorm2d(2*c1)
        self.conv3 = nn.Conv2d(2*c1,c2,kernel_size=1)
        self.bn3=nn.BatchNorm2d(c2)
        #自适应权重
        self.weight1 = AdaptiveWeight(1)
        self.weight2 = AdaptiveWeight(1)
        self.adaptive_flag=False
        if c1==c2:
            self.adaptive_flag=True
        #CA
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)),inplace=True)
        #高频信息引入
        downx1 = self.down(x1)
        hf_info = x1 - F.interpolate(downx1,size=x.shape[-2:],mode='bilinear',align_corners=True)
        x1=torch.cat((x1,hf_info),dim=1)
        x2 = F.relu(self.bn2(self.conv2(x1)),inplace=True) 
        x3 =F.relu(self.bn3(self.conv3(x2)),inplace=True)
        out=x3
        if self.adaptive_flag:
            out=self.weight1(x)+self.weight2(x3)
        return out

#外部注意力封装
#MLP  B,C,H,W
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features,1, bias=True)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features,1,bias=True)
    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class External_block(nn.Module):
    def __init__(self,d_model,S):
        super().__init__()
        self.ea = ExternalAttention(d_model=d_model,S=S)
        self.norm=nn.LayerNorm(d_model)
        self.w= nn.Parameter(torch.FloatTensor(2))#权重设置
        self.mlp = Mlp(in_features=d_model, hidden_features=d_model, act_layer=nn.ReLU, drop=0.)
        

    def forward(self, x):
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        B,C,H,W=x.shape
        x = x.flatten(2).transpose(1, 2) #B,C,H,W->B,N,C
        x = w1*x + w2*self.ea(self.norm(x))        
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2) #B,N,C->B,C,H,W
        x = x + self.mlp(x)
        return x
    
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 3, s, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, c1, c2, s=1, downsample=None, no_relu=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c1, c2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.conv2 = nn.Conv2d(c2, c2, 3, s, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.conv3 = nn.Conv2d(c2, c2 * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2 * self.expansion)
        self.downsample = downsample
        self.no_relu = no_relu

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        out += identity
        return out if self.no_relu else F.relu(out)


class ConvBN(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Conv2BN(nn.Sequential):
    def __init__(self, c1, ch, c2, k, s=1, p=0):
        super().__init__(
            nn.Conv2d(c1, ch, k, s, p, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.Conv2d(ch, c2, k, s, p, bias=False),
            nn.BatchNorm2d(c2)
        )


class Stem(nn.Sequential):
    def __init__(self, c1, c2):
        super().__init__(
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True),
            nn.Conv2d(c2, c2, 3, 2, 1),
            nn.BatchNorm2d(c2),
            nn.ReLU(True)
        )


class Scale(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.AvgPool2d(k, s, p),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ScaleLast(nn.Sequential):
    def __init__(self, c1, c2, k):
        super().__init__(
            nn.AdaptiveAvgPool2d(k),
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, 1, bias=False)
        )


class ConvModule(nn.Sequential):
    def __init__(self, c1, c2, k, s=1, p=0):
        super().__init__(
            nn.BatchNorm2d(c1),
            nn.ReLU(True),
            nn.Conv2d(c1, c2, k, s, p, bias=False)
        )


class DAPPM(nn.Module):
    def __init__(self, c1, ch, c2):
        super().__init__()
        self.scale1 = Scale(c1, ch, 5, 2, 2)
        self.scale2 = Scale(c1, ch, 9, 4, 4)
        self.scale3 = Scale(c1, ch, 17, 8, 8)
        self.scale4 = ScaleLast(c1, ch, 1)
        self.scale0 = ConvModule(c1, ch, 1)
        self.process1 = ConvModule(ch, ch, 3, 1, 1)
        self.process2 = ConvModule(ch, ch, 3, 1, 1)
        self.process3 = ConvModule(ch, ch, 3, 1, 1)
        self.process4 = ConvModule(ch, ch, 3, 1, 1)
        self.compression = ConvModule(ch * 5, c2, 1)
        self.shortcut = ConvModule(c1, c2, 1)

    def forward(self, x: Tensor) -> Tensor:
        outs = [self.scale0(x)]
        outs.append(self.process1(
            (F.interpolate(self.scale1(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        outs.append(self.process2(
            (F.interpolate(self.scale2(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        outs.append(self.process3(
            (F.interpolate(self.scale3(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        outs.append(self.process4(
            (F.interpolate(self.scale4(x), size=x.shape[-2:], mode='bilinear', align_corners=False) + outs[-1])))
        out = self.compression(torch.cat(outs, dim=1)) + self.shortcut(x)
        return out


class SegHead(nn.Module):
    def __init__(self, c1, ch, c2, scale_factor=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(c1)
        self.conv1 = nn.Conv2d(c1, ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, c2, 1)
        self.scale_factor = scale_factor
    def forward(self, x: Tensor) -> Tensor:

        x = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(x)))
        if self.scale_factor is not None:
            H, W = x.shape[-2] * self.scale_factor, x.shape[-1] * self.scale_factor
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
        return out
class final_model(nn.Module):
    def __init__(self, backbone: str = None, num_classes: int = 19) -> None:
        super().__init__()
        planes, spp_planes, head_planes = [32, 64, 128, 256, 512], 128, 64

        self.conv1 = Stem(3, planes[0])
        self.layer1 = self._make_layer(BasicBlock, planes[0], planes[0], 2)
        self.layer2 = self._make_layer(BasicBlock, planes[0], planes[1], 2, 2)

        #mobilevit  setting
        mv2_exp_mult = 2
        ffn_multiplier = 2
        attn_dim = [128, 192, 256]
        patch_size=(2,2)

        #64 => 128
        self.layer3 = nn.Sequential(
            InvertedResidual(planes[1], planes[2], stride=2, expand_ratio=mv2_exp_mult),
        )
        self.layer3_MobileViTBlockv3_v2= MobileViTBlockv3_v2(planes[2], attn_dim[0], ffn_multiplier,  attn_blocks=1, patch_size=patch_size)
        self.layer4 = nn.Sequential(
            InvertedResidual(planes[2], planes[3], stride=2, expand_ratio=mv2_exp_mult),#128->512->256
            External_block(d_model=256,S=64)
        )
        self.layer5 = nn.Sequential(
            InvertedResidual(planes[3], planes[3], stride=1, expand_ratio=mv2_exp_mult),

        )
        self.layer5_MobileViTBlockv3_v2 =MobileViTBlockv3_v2(planes[3], attn_dim[2], ffn_multiplier, attn_blocks=1,
                                patch_size=patch_size)
        self.layer3_ = self._make_layer(Block, planes[1], planes[1], 2)
        self.layer4_ = self._make_layer(Block, planes[1], planes[1], 2)
        self.layer5_ = self._make_layer(Block, planes[1], planes[2], 1)

        self.compression3 = ConvBN(planes[2], planes[1], 1)
        self.compression4 = ConvBN(planes[3], planes[1], 1)
        
        self.down3 = ConvBN(planes[1], planes[2], 3, 2, 1)
        self.down4 = ConvBN(planes[1], planes[2], 3, 2, 1)
        self.down5 = Conv2BN(planes[2], planes[2], planes[3], 3, 2, 1)
        self.spp = DAPPM(planes[-2], spp_planes, planes[2])
        self.seghead_extra = SegHead(planes[1], head_planes, num_classes, 8)
        self.final_layer = SegHead(planes[2], head_planes, num_classes, 8)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def init_pretrained(self, pretrained: str = None) -> None:
        if pretrained:
            self.load_state_dict(torch.load(pretrained, map_location='cpu')['model'], strict=False)

    def _make_layer(self, block, inplanes, planes, depths, s=1) -> nn.Sequential:
        downsample = None
        if inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion, 1, s, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        if block.__name__=='Block':
            layers = [block(inplanes, planes, s)]
        else:
            layers = [block(inplanes, planes, s, downsample)]
        inplanes = planes * block.expansion

        for i in range(1, depths):
            if i == depths - 1:
                if block.__name__=='Block':
                    layers.append(block(inplanes, planes))
                else:
                    layers.append(block(inplanes, planes, no_relu=True))                
            else:
                layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        B = x.shape[0]
        H, W = x.shape[-2] // 8, x.shape[-1] // 8
        layers = []

        x = self.conv1(x)
        x = self.layer1(x)
        layers.append(x)

        x = self.layer2(F.relu(x))
        layers.append(x)
        x_ = self.layer3_(F.relu(layers[1]))
        # transformer branch3
        x = self.layer3(x)
        x = self.layer3_MobileViTBlockv3_v2(x,self.down3(x_))
        layers.append(x)

        x_ = x_ + F.interpolate(self.compression3(F.relu(layers[2])), size=(H, W), mode='bilinear', align_corners=False)
        if self.training: x_aux = self.seghead_extra(x_)

        # cnn branch4
        x_ = self.layer4_(F.relu(x_))

        x = x + self.down4(F.relu(x_))

        # transformer branch4

        x=self.layer4(x)
        layers.append(x)
        x_ = x_ + F.interpolate(self.compression4(F.relu(layers[3])), size=(H, W), mode='bilinear', align_corners=False)
        # cnn branch5
        x_ = self.layer5_(F.relu(x_))

        # transformer branch5
        x = self.layer5(x)
        x = self.layer5_MobileViTBlockv3_v2(x,self.down5(x_))
        x = F.interpolate(self.spp(x), size=(H, W), mode='bilinear', align_corners=False)
        x_ = self.final_layer(x + x_)
        return (x_, x_aux) if self.training else x_
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    model = final_model(num_classes=4).cuda()
    input=torch.randn(1,3,1600,256).cuda()
    print(model)
    model.eval()
    print(model)
    # # print(model)
    # model(input)
    flops, params = profile(model, inputs=(input,))
    print("FLOPs=", str(flops/1e9) + '{}'.format("G"))
    print("params=", str(params/1e6) + '{}'.format("M"))
    model.train(False)
    model.eval()
    input=torch.randn(1,3,1600,256).cuda()
    warm_iter=300
    iteration=1000
    print('=========Speed Testing=========')
    fps_time=[]
    for _ in range(iteration): #iteration=20
        if _<warm_iter:
            model(input)
        else:
            start=time.time()
            model(input)
            end=time.time()
            fps_time.append(end-start)
    time_sum = 0
    for i in fps_time:
        time_sum += i
    print("FPS: %f"%(1.0/(time_sum/len(fps_time))))