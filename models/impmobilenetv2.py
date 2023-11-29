import torch
import torch.nn as nn
import os
import math
from novelStructure.HetConv import HetConv
from novelStructure.acon import MetaAconC
from novelStructure.StemBlock import StemBlock
from tensorboardX import SummaryWriter
import datetime

# 表示2D卷积的 3*3 和 1*1 的分布情况
p = 4
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

# class swish(nn.Module):
#     def forward(self, x):
#         return x * torch.sigmoid(x)
#         return x * y

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.relu = h_swish()

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)

        y = identity * x_w * x_h

        return y


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


#  最简单最易实现的SE模块 https://www.cnblogs.com/pprp/p/12128520.html
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# 定义MobilenetV2 模型结构，目的是方便进行结构上的优化与修改
def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
        # MetaAconC(oup)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
        # MetaAconC(oup)

    )


def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x * 1. / divisible_by) * divisible_by)



class InvertedResidual_MetaAcon(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_MetaAcon, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),

            )

        else:
            if stride == 1:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),

                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),

                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),

                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    MetaAconC(hidden_dim),

                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU6(inplace=True),
                )




    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




class InvertedResidual_HetConv(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_HetConv, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw

                HetConv(hidden_dim, hidden_dim, p=4),
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),

            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),

                # dw
                HetConv(hidden_dim, hidden_dim, p=4),
                # nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class InvertedResidual_CA(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_CA, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),


                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # coordinate attention
                CoordAtt(hidden_dim, hidden_dim),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)




class InvertedResidual_shuffle(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_shuffle, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        # stride == 1 时在累加残差
        # self.use_res_connect = self.stride == 1and inp == oup
        self.use_res_connect = self.stride == 1 and expand_ratio != 1

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # pw-linear　　　　　　　　　　　　　　　　　　　　　　　　
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if self.use_res_connect:

            #失败
            # x1 = x[:, :(x.shape[1] // 2), :, :]
            # x2 = x[:, (x.shape[1] // 2):, :, :]
            # x3 = self.conv(x2)
            # x4 = self._concat(x1, x3)
            # return x4

            # acc = 0.8566
            # x1 = x[:, :(x.shape[1] // 2), :, :]
            # x2 = self.conv(x)
            # x3 = x2[:, (x2.shape[1] // 2):, :, :]
            # return self._concat(x1, x3)

            out = self.conv(x)
            oup = self._concat(x, out)
            return oup
        else:
            op = self.conv(x)
            return op

def channel_split(x, split):
    """split a tensor into two pieces along channel dimension
    Args:
        x: input tensor
        split:(int) channel size for each pieces
    """
    assert x.size(1) == split * 2
    return torch.split(x, split, dim=1)

def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    # b, c, h, w =======>  b, g, c_per, h, w
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batch_size, -1, height, width)
    return x



class InvertedResidual_EE(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_EE, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup
        self.g = 8
        self.expand_ratio = expand_ratio

        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=self.g, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.pw = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=self.g, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
            self.dw = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=self.g, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            )
            self.pw_linear = nn.Sequential(
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, groups=self.g, bias=False),
                nn.BatchNorm2d(oup)
            )




    def forward(self, x):
        tem = x
        if self.expand_ratio == 1:
            x = self.conv(x)
        else:
            x = self.pw(x)
            x = channel_shuffle(x, self.g)
            x = self.dw(x)
            x = self.pw_linear(x)

        if self.use_res_connect:
            return tem + x
        else:
            return x




class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),


                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),

                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



# # baseline
class MobileNetV2(nn.Module):

    def __init__(self, input_size=32, class_num=10,  width_mult=1.):
        super(MobileNetV2, self).__init__()
        # 原始 BottleNeck
        block = InvertedResidual

        # BottleNeck 使用metaAcon
        # block = InvertedResidual_MetaAcon

        # BottleNeck 使用hetconv
        # block = InvertedResidual_HetConv

        # block = InvertedResidual_shuffle

        # block = InvertedResidual_CA

        # block = InvertedResidual_EE

         ########################  改动1 将输入的 通道数由 32 改为 16############################################
        input_channel = input_size
        last_channel = 1280


        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        self.interverted_residual_setting = interverted_residual_setting

        # building first layer
        assert input_size % 32 == 0
        input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
        self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
        # add stemblock main-2跑的这个
        # self.features = [StemBlock(3, input_channel)]

        self.features = [conv_bn(3, input_channel, 2)]
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = make_divisible(c * width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))

                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # 全局平均池化需要添加。确保最后输出的　ｋernel size 是１＊１
        self.avg = nn.AdaptiveAvgPool2d(1)

        # building classifier
        self.classifier = nn.Linear(self.last_channel, class_num)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #添加全局平均池化
        x = self.avg(x)
        x = x.mean(3).mean(2)
        x = self.classifier(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()











def IR_Conv1(inp, oup, hidden_dim=1):
    return torch.nn.Sequential(
                # dw
                nn.Conv2d(inp*hidden_dim, inp*hidden_dim, 3, 1, 1, groups=inp*hidden_dim, bias=False),
                nn.BatchNorm2d(inp*hidden_dim),
                nn.ReLU6(inplace=True),

                # pw-linear
                nn.Conv2d(inp*hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )


def IR_Conv2(inp, oup, stride, hidden_dim=6):
    return torch.nn.Sequential(
        # pw
        torch.nn.Conv2d(inp, inp * hidden_dim, 1, 1, 0, bias=False),
        torch.nn.BatchNorm2d(inp * hidden_dim),
        torch.nn.ReLU6(inplace=True),
        # dw
        torch.nn.Conv2d(inp * hidden_dim, inp * hidden_dim, 3, stride, 1, groups=inp * hidden_dim, bias=False),
        torch.nn.BatchNorm2d(inp * hidden_dim),
        torch.nn.ReLU6(inplace=True),
        # pw-linear
        torch.nn.Conv2d(inp * hidden_dim, oup, 1, 1, 0, bias=False),
        torch.nn.BatchNorm2d(oup),
    )


if __name__ == '__main__':
    net = MobileNetV2()
    print('mobilenetv2:\n', net)
    from thop import clever_format, profile
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(net, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("Total  trainable flops:", flops)
    print("Total  trainable params: ", params)



















# 5-14　号代码
# class MobileNetV2(nn.Module):
#
#     def __init__(self, n_class=10, input_size=32, width_mult=1.):
#         super(MobileNetV2, self).__init__()
#
#
#          ########################  改动1 将输入的 通道数由 32 改为 16############################################
#         input_channel = 32
#         last_channel = 1280
#
#         # building first layer
#         assert input_size % 16 == 0
#         input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
#         self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
#
#         # #  方式　２　 acc =  0.8743  但是模型转换失败
#         self.firstConv = torch.nn.Sequential(
#             torch.nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
#             torch.nn.BatchNorm2d(input_channel),
#             torch.nn.ReLU6(inplace=True)
#         )
#         ########################################################################################
#
#
#         inp = input_channel
#         oup = 16
#         hidden_dim = 1
#         self.conv1 = IR_Conv1(inp, oup, hidden_dim)
#         ########################################################################################
#         inp1_1 = 16
#         oup1_1 = 24
#         hidden_dim = 6
#         self.conv2_1 = IR_Conv2(inp1_1, oup1_1, 2, hidden_dim)
#
#         inp2_2 = 24
#         oup2_2 = 24
#         self.conv2_2 = IR_Conv2(inp2_2, oup2_2, 1, hidden_dim)
#         ########################################################################################
#         inp3_1 = 24
#         oup3_1 = 32
#         self.conv3_1 = IR_Conv2(inp3_1, oup3_1, 2, hidden_dim)
#
#         inp3_2 = 32
#         oup3_2 = 32
#         self.conv3_2 = IR_Conv2(inp3_2, oup3_2, 1, hidden_dim)
#         inp3_3 = 32
#         oup3_3 = 32
#         self.conv3_3 = IR_Conv2(inp3_3, oup3_3, 1, hidden_dim)
#         self.GFF3 = nn.Conv2d(inp3_3*2, oup3_3, 1, padding=0, stride=1)
#         ########################################################################################
#         inp4_1 = 32
#         oup4_1 = 64
#         self.conv4_1 = IR_Conv2(inp4_1, oup4_1, 2, hidden_dim)
#
#         inp4_2 = 64
#         oup4_2 = 64
#         self.conv4_2 = IR_Conv2(inp4_2, oup4_2, 1, hidden_dim)
#
#         inp4_3 = 64
#         oup4_3 = 64
#         self.conv4_3 = IR_Conv2(inp4_3, oup4_3, 1, hidden_dim)
#         inp4_4 = 64
#         oup4_4 = 64
#         self.conv4_4 = IR_Conv2(inp4_4, oup4_4, 1, hidden_dim)
#         self.GFF4 = nn.Conv2d(inp4_4*3, oup4_4, 1, padding=0, stride=1)
#         ########################################################################################
#         inp5_1 = 64
#         oup5_1 = 96
#         self.conv5_1 = IR_Conv2(inp5_1, oup5_1, 1, hidden_dim)
#         inp5_2 = 96
#         oup5_2 = 96
#         self.conv5_2 = IR_Conv2(inp5_2, oup5_2, 1, hidden_dim)
#         inp5_3 = 96
#         oup5_3 = 96
#         self.conv5_3 = IR_Conv2(inp5_3, oup5_3, 1, hidden_dim)
#         self.GFF5 = nn.Conv2d(inp5_3*2, oup5_3, 1, padding=0, stride=1)
#
#         ########################################################################################
#         inp6_1 = 96
#         oup6_1 = 160
#         self.conv6_1 = IR_Conv2(inp6_1, oup6_1, 2, hidden_dim)
#         inp6_2 = 160
#         oup6_2 = 160
#         self.conv6_2 = IR_Conv2(inp6_2, oup6_2, 1, hidden_dim)
#         inp6_3 = 160
#         oup6_3 = 160
#         self.conv6_3 = IR_Conv2(inp6_3, oup6_3, 1, hidden_dim)
#         self.GFF6 = nn.Conv2d(inp6_3*2, oup6_3, 1, padding=0, stride=1)
#
#         ########################################################################################
#         inp7_1 = 160
#         oup7_1 = 320
#         self.conv7_1 = IR_Conv2(inp7_1, oup7_1, 1, hidden_dim)
#         ########################################################################################
#
#         last_inp = 320
#
#         self.lastConv = torch.nn.Sequential(
#             torch.nn.Conv2d(last_inp, self.last_channel, 1, 1, 0, bias=False),
#             torch.nn.BatchNorm2d(self.last_channel),
#             torch.nn.ReLU6(inplace=True)
#         )
#
#         self.classifier = torch.nn.Linear(self.last_channel, n_class)
#
#         self._initialize_weights()
#
#     @staticmethod
#     def _concat(x, out):
#         # concatenate along channel axis
#         return torch.cat((x, out), 1)
#
#     def forward(self, x):
#
#         x = self.firstConv(x)
#
#         #block1
#         bx1 = self.conv1(x)
#
#
#         #block2
#         x = self.conv2_1(bx1)
#         bx2 = x+self.conv2_2(x)
#
#
#         #block3
#         x = self.conv3_1(bx2)
#         x = x+self.conv3_2(x)
#         bx3 = x + self.conv3_3(x)
#         bx3 = self.GFF3(torch.cat((x, bx3), 1))
#
#
#         #block4
#         x = self.conv4_1(bx3)
#         x1 = x+self.conv4_2(x)
#         x2 = x1 + self.conv4_3(x1)
#         bx4 = x2 + self.conv4_4(x2)
#         bx4 = self.GFF4(torch.cat((x1, x2, bx4), 1))
#
#
#         #block5
#         x = self.conv5_1(bx4)
#         x1 = x+self.conv5_2(x)
#         bx5 = x+self.conv5_3(x1)
#         bx5 = self.GFF5(torch.cat((x1, bx5), 1))
#
#
#         #block6
#         x = self.conv6_1(bx5)
#         x1 = x+self.conv6_2(x)
#         bx6 = x+self.conv6_3(x1)
#         bx6 = self.GFF6(torch.cat((x1, bx6), 1))
#
#
#         #block7
#         bx7 = self.conv7_1(bx6)
#
#
#         x = self.lastConv(bx7)
#
#
#
#         x = x.mean(3).mean(2)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()


# # 5-13
# class MobileNetV2(nn.Module):
#
#     def __init__(self, n_class=10, input_size=32, width_mult=1.):
#         super(MobileNetV2, self).__init__()
#
#
#          ########################  改动1 将输入的 通道数由 32 改为 16############################################
#         input_channel = 32
#         last_channel = 1280
#
#         # building first layer
#         assert input_size % 16 == 0
#         input_channel = make_divisible(input_channel * width_mult)  # first channel is always 32!
#         self.last_channel = make_divisible(last_channel * width_mult) if width_mult > 1.0 else last_channel
#
#         # #  方式　２　 acc =  0.8743  但是模型转换失败
#         self.firstConv = torch.nn.Sequential(
#             torch.nn.Conv2d(3, 32, 3, 2, 1, bias=False),
#             torch.nn.BatchNorm2d(32),
#             torch.nn.ReLU6(inplace=True)
#         )
#         ########################################################################################
#         inp = input_channel
#         oup = 16
#         hidden_dim = 1
#         self.conv1 = IR_Conv1(inp, oup, hidden_dim)
#         ########################################################################################
#         inp1_1 = 16
#         oup1_1 = 24
#         hidden_dim = 6
#         self.conv2_1 = IR_Conv2(inp1_1, oup1_1, 2, hidden_dim)
#
#         inp2_2 = 24
#         oup2_2 = 24
#         self.conv2_2 = IR_Conv2(inp2_2, oup2_2, 1, hidden_dim)
#         ########################################################################################
#         inp3_1 = 24
#         oup3_1 = 32
#         self.conv3_1 = IR_Conv2(inp3_1, oup3_1, 2, hidden_dim)
#
#         inp3_2 = 32
#         oup3_2 = 32
#         self.conv3_2 = IR_Conv2(inp3_2, oup3_2, 1, hidden_dim)
#         inp3_3 = 32
#         oup3_3 = 32
#         self.conv3_3 = IR_Conv2(inp3_3, oup3_3, 1, hidden_dim)
#         ########################################################################################
#         inp4_1 = 32
#         oup4_1 = 64
#         self.conv4_1 = IR_Conv2(inp4_1, oup4_1, 2, hidden_dim)
#
#         inp4_2 = 64
#         oup4_2 = 64
#         self.conv4_2 = IR_Conv2(inp4_2, oup4_2, 1, hidden_dim)
#
#         inp4_3 = 64
#         oup4_3 = 64
#         self.conv4_3 = IR_Conv2(inp4_3, oup4_3, 1, hidden_dim)
#         inp4_4 = 64
#         oup4_4 = 64
#         self.conv4_4 = IR_Conv2(inp4_4, oup4_4, 1, hidden_dim)
#         ########################################################################################
#         inp5_1 = 64
#         oup5_1 = 96
#         self.conv5_1 = IR_Conv2(inp5_1, oup5_1, 1, hidden_dim)
#         inp5_2 = 96
#         oup5_2 = 96
#         self.conv5_2 = IR_Conv2(inp5_2, oup5_2, 1, hidden_dim)
#         inp5_3 = 96
#         oup5_3 = 96
#         self.conv5_3 = IR_Conv2(inp5_3, oup5_3, 1, hidden_dim)
#         ########################################################################################
#         inp6_1 = 96
#         oup6_1 = 160
#         self.conv6_1 = IR_Conv2(inp6_1, oup6_1, 2, hidden_dim)
#         inp6_2 = 160
#         oup6_2 = 160
#         self.conv6_2 = IR_Conv2(inp6_2, oup6_2, 1, hidden_dim)
#         inp6_3 = 160
#         oup6_3 = 160
#         self.conv6_3 = IR_Conv2(inp6_3, oup6_3, 1, hidden_dim)
#
#         ########################################################################################
#         inp7_1 = 160
#         oup7_1 = 320
#         self.conv7_1 = IR_Conv2(inp7_1, oup7_1, 1, hidden_dim)
#         ########################################################################################
#
#         last_inp = 320
#
#         self.lastConv = torch.nn.Sequential(
#             torch.nn.Conv2d(last_inp, self.last_channel, 1, 1, 0, bias=False),
#             torch.nn.BatchNorm2d(self.last_channel),
#             torch.nn.ReLU6(inplace=True)
#         )
#
#         self.classifier = torch.nn.Linear(self.last_channel, n_class)
#
#         self._initialize_weights()
#
#     @staticmethod
#     def _concat(x, out):
#         # concatenate along channel axis
#         return torch.cat((x, out), 1)
#
#     def forward(self, x):
#
#         x = self.firstConv(x)
#
#         #block1
#         bx1 = self.conv1(x)
#
#
#         #block2
#         x = self.conv2_1(bx1)
#         bx2 = x+self.conv2_2(x)
#
#
#         #block3
#         x = self.conv3_1(bx2)
#         x1 = x+self.conv3_2(x)
#         bx3 = x + x1 + self.conv3_3(x1)
#
#
#         #block4
#         x = self.conv4_1(bx3)
#         x1 = x+self.conv4_2(x)
#         x2 = x + x1 + self.conv4_3(x1)
#         bx4 = x + x1 + x2 +self.conv4_4(x2)
#
#
#         #block5
#         x = self.conv5_1(bx4)
#         x1 = x+self.conv5_2(x)
#         bx5 = x+x1+self.conv5_3(x1)
#
#
#         #block6
#         x = self.conv6_1(bx5)
#         x1 = x+self.conv6_2(x)
#         bx6 = x+x1+self.conv6_3(x1)
#
#
#         #block7
#         bx7 = self.conv7_1(bx6)
#
#         x = self.lastConv(bx7)
#
#
#
#         x = x.mean(3).mean(2)
#         x = self.classifier(x)
#         return x
#
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 n = m.weight.size(1)
#                 m.weight.data.normal_(0, 0.01)
#                 m.bias.data.zero_()




