import torch
import torch.nn as nn

# 定义Net 更改网络最后两层的结构，使得去掉分类器，输出全连接层
#squeezeNet -2    MobileNetv1 －２   MobileNetV2 -1    MobileNetV3 -1    ShuffleNetv1    shuffleNetv2-3  MyNet -1
class Net(nn.Module):
    def __init__(self, model):
        super(Net, self).__init__()
        # -1表示去掉model的最后一层分类器
        self.mobile_layer = nn.Sequential(*list(model.children())[:-1])
        # self.Linear_layer = nn.Conv2d(512, 32, 1, 4, 0, bias=False)

    def forward(self, x):
        x = self.mobile_layer(x)
        # 将前面多维度的tensor展平成一维,-1指在不告诉函数有多少列
        # x = x.view(x.size(0), -1)
        # x = self.Linear_layer(x)
        # return feature, x
        return x


