# ***************************一些必要的包的调用********************************
import torch.nn.functional as F
import torch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from tqdm import tqdm

# ***************************初始化一些函数********************************
# torch.cuda.set_device(gpu_id)#使用GPU
learning_rate = 0.0001  # 学习率的设置

# *************************************数据集的设置****************************************************************************
root ='/media/llj0571/OS/SUN397/Partitions/'  # 数据集的地址


# 定义读取文件的格式
def default_loader(path):
    img = Image.open(path)
    if img.mode is not "RGB":
        img = img.convert("RGB")
    return img


class SUN397Dataset(Dataset):
    # 创建自己的类： MyDataset,这个类是继承的torch.utils.data.Dataset
    # **********************************  #使用__init__()初始化一些需要传入的参数及数据集的调用**********************
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        super(SUN397Dataset, self).__init__()
        # 对继承自父类的属性进行初始化
        train_file_name = os.path.join('/media/llj0571/OS/SUN397/Partitions/', txt)

        fh = open(train_file_name, 'r')
        imgs = []
        # 按照传入的路径和txt文本参数，以只读的方式打开这个文本
        for line in fh:  # 迭代该列表#按行循环txt文本中的内
            line = line.strip('\n')
            line = line.rstrip('\n')
            # 删除 本行string 字符串末尾的指定字符，这个方法的详细介绍自己查询python
            # words = line.split('/')
            split_indices = [i for i, letter in enumerate(line) if letter == '/']
            className = line[split_indices[0]:split_indices[len(split_indices)-1]]
            # 根据classname 去　scene_names.txt 中查找对应的Label
            class_file_name = os.path.join('/media/llj0571/OS/SUN397/Partitions/', "scene_names.txt")


            with open(class_file_name) as class_file:
                for lineinx in class_file:
                    line_first = lineinx.split()[0]
                    line_label = lineinx.split()[1]
                    if className == line_first:
                        imgs.append((line, int(line_label)))
                        break

        # 很显然，根据我刚才截图所示txt的内容，words[0]是图片信息，words[1]是lable
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    # *************************** #使用__getitem__()对数据进行预处理并返回想要的信息**********************
    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        fn, label = self.imgs[index]
        # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        # img = self.loader(fn)
        imgroot = '/media/llj0571/OS/SUN397/SUN397'
        # print('哪张图片错误了：'+imgroot+fn)
        # 按照路径读取图片
        img = self.loader(imgroot+fn)
        # print(img)

        if self.transform is not None:
            img = self.transform(img)
        # 数据标签转换为Tensor
        return img, label
       # return回哪些内容，那么我们在训练时循环读取每个batch时，就能获得哪些内容


    # **********************************  #使用__len__()初始化一些需要传入的参数及数据集的调用**********************
    def __len__(self):
      # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
       return len(self.imgs)





if __name__ == '__main__':

    transformations = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = SUN397Dataset(txt='/media/llj0571/OS/SUN397/Partitions/train.txt', transform=transformations)
    # for batch_idx, (data, target) in tqdm(enumerate(train_data), total=len(train_data)):
    #     data, target = data, target
    #     print(data)
    test_data = SUN397Dataset(txt='/media/llj0571/OS/SUN397/Partitions/test.txt', transform=transformations)


