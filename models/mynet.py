import torch
import torch.nn as nn
import torch.nn.functional as F



def channel_shuffle(x, groups):
    """channel shuffle operation
    Args:
        x: input tensor
        groups: input branch number
    """

    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels // groups)

    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class baseBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super().__init__()

        hidden_dim = int(in_channel * expand_ratio)
        self.hiddendim = hidden_dim
        self.use_res_connect = in_channel == out_channel

        self.basepw = nn.Sequential(
            # pw
            nn.Conv2d(in_channel, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.basedw = nn.Sequential(
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.baseExpand = nn.Sequential(
            # pw-linear
            nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        x = self.basepw(x)
        x = channel_shuffle(x, 1)
        x = self.basedw(x)
        x = self.baseExpand(x)
        return x


class myBlock(nn.Module):

    def __init__(self, in_channel, out_channel, stride, expand_ratio, group):
        super().__init__()

        hidden_dim = int(in_channel * expand_ratio)
        self.group = group
        self.use_res_connect = in_channel == out_channel

        # out_channel = int(out_channel * expand_ratio)

        # self.shortcut = nn.Sequential()

        self.pw = nn.Sequential(
            # pw
            nn.Conv2d(in_channel, hidden_dim, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )
        self.dw = nn.Sequential(
            # dw
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=group, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        )

        self.expand = nn.Sequential(
            # pw-linear
            nn.Conv2d(hidden_dim, out_channel, 1, 1, 0, groups=group, bias=False),
            nn.BatchNorm2d(out_channel)
        )

        # pw-linear
        # self.expand_1x1 = nn.Sequential(
        #     # pw-linear
        #     nn.Conv2d(hidden_dim, int(out_channel / 2), 1),
        #     nn.BatchNorm2d(int(out_channel / 2)),
        #     nn.ReLU(inplace=True)
        # )
        # self.expand_3x3 = nn.Sequential(
        #     nn.Conv2d(hidden_dim, int(out_channel / 2), 3, padding=1),
        #     nn.BatchNorm2d(int(out_channel / 2)),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        # tem = x
        x = self.pw(x)
        # f1
        # x = channel_shuffle(x, self.group)

        x = self.dw(x)

        # x = channel_shuffle(x, self.group)
        # f２
        x = self.expand(x)
        return x



class MyNet(nn.Module):

    def __init__(self, input_size=32, class_num=10, alpha=1.):
        super(MyNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, input_size, 3, padding=1),
            nn.BatchNorm2d(input_size),
            nn.ReLU(inplace=True)
        )

        # group = 8
        # # 无残差，　无多次迭代ＢLOCK
        # self.block1 = myBlock(input_size, 16, 1, 1, group)
        # self.block2 = myBlock(16, 24, 2, 6, group)
        # self.block3 = myBlock(24, 32, 2, 6, group)
        # self.block4 = myBlock(32, 64, 2, 6, group)
        # self.block5 = myBlock(64, 96, 1, 6, group)
        # self.block6 = myBlock(96, 160, 2, 6, group)
        # self.block7 = myBlock(160, 320, 1, 6, group)

        #　无多次迭代ＢLOCK　　cnn \ cnn+cs
        self.block1 = baseBlock(input_size, 16, 1, 1)
        self.block2 = baseBlock(16, 24, 2, 6)
        self.block3 = baseBlock(24, 32, 2, 6)
        self.block4 = baseBlock(32, 64, 2, 6)
        self.block5 = baseBlock(64, 96, 1, 6)
        self.block6 = baseBlock(96, 160, 2, 6)
        self.block7 = baseBlock(160, 320, 1, 6)

        # group = 16
        # self.block1 = myBlock(input_size, 16, 1, 1, group)
        # self.block2 = myBlock(16, 32, 2, 6, group)
        # self.block3 = myBlock(32, 48, 2, 6, group)
        # self.block4 = myBlock(48, 64, 2, 6, group)
        # self.block5 = myBlock(64, 96, 1, 6, group)
        # self.block6 = myBlock(96, 160, 2, 6, group)
        # self.block7 = myBlock(160, 320, 1, 6, group)

        # 有残差结构
        # self.block1 = myBlock(input_size, 16, 1, 1, group)
        # self.block2 = self._make_stage(2, 16, 24, 2, 6, group)
        # self.block3 = self._make_stage(3, 24, 32, 2, 6, group)
        # self.block4 = self._make_stage(4, 32, 64, 2, 6, group)
        # self.block5 = self._make_stage(3, 64, 96, 1, 6, group)
        # self.block6 = self._make_stage(3, 96, 160, 2, 6, group)
        # self.block7 = myBlock(1, 60, 320, 1, 6, group)

        self.last = nn.Sequential(
            nn.Conv2d(320, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        # 全局平均池化需要添加。因为　最后输出的　ｋernel size 不是１＊１
        self.avg = nn.AdaptiveAvgPool2d(1)
        # self.avg = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(1280, class_num)



    def forward(self, x):
         x = self.stem(x)
         x = self.block1(x)
         x = self.block2(x)
         x = self.block3(x)
         x = self.block4(x)
         x = self.block5(x)
         x = self.block6(x)
         x = self.block7(x)

         x = self.last(x)

         x = self.avg(x)
         x = x.mean(3).mean(2)
         x = self.classifier(x)
         return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t, group):

        layers = []
        layers.append(myBlock(in_channels, out_channels, stride, t, group))

        while repeat - 1:
            layers.append(myBlock(out_channels, out_channels, 1, t, group))
            repeat -= 1

        return nn.Sequential(*layers)



def mynet(input_size=32, class_num=10, alpha=1):
    return MyNet(input_size, class_num, alpha)

if __name__ == '__main__':
    from thop import clever_format, profile
    # input = torch.randn(1, 3, 32, 32)
    # net = mynet(input)
    # print('mynet:\n', net)

    net = MyNet()
    print(net)
    input = torch.randn(1, 3, 32, 32)
    output = net(input)

    print(output)
    flops, params = profile(net, inputs=(input,), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print("Total  trainable flops:", flops)
    print("Total  trainable params: ", params)
