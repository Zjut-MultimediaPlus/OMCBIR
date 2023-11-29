import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import os
import numpy as np
import math
from PIL import Image
from models import *



print('读取1000张测试图片，提取特征，存入数据库===================================================')
count = 0
# 2、读取本地图片，提取特征，存入数据库
# dir_path = "/home/zxq/code/testImage/"
dir_path = "/home/llj0571/Desktop/code/datasets/retrivalDBImage/"

#0.2049
# model = torch.load("./checkpoints/MobileNet/v1/model-cpu.pt")

#0.1846
# model = torch.load("./checkpoints/MobileNetV2/baseline/model-cpu.pt")

#0.1779
# model = torch.load("./checkpoints/MobileNetV3/v3/model-cpu.pt")

#0.2176
model = torch.load("./checkpoints/MyNet/model/model-cpu.pt")

#0.350
# model = torch.load("./checkpoints/ShuffleNetV2/v2/model-cpu.pt")

#0.2360
# model = torch.load("./checkpoints/SqueezeNet/model/model-cpu.pt")





print(model)
model.eval()

feature_list = []
for filename in os.listdir(dir_path):  # listdir的参数是文件夹的路径
    print(filename)
    image = Image.open(dir_path + filename)  # 图片发在了build文件夹下
    image = image.resize((32, 32), Image.ANTIALIAS)
    image = np.asarray(image)
    image = image / 255
    image = torch.Tensor(image).unsqueeze_(dim=0)
    image = image.permute((0, 3, 1, 2)).float()
    outputs = model(image)
    outputs = outputs.data.numpy()
    outputs = outputs[0]
    # 不使用数据库，直接放到数组里操作，实现检索
    dict = {"imgName": filename, "feature": outputs}
    feature_list.append(dict)

# 测试单个图片，服务端与移动端特征值的差别
ap_path = "/home/llj0571/Desktop/code/datasets/retrivalImage/lable9/test_9_11.jpg"
searImg = Image.open(ap_path)
searImg = searImg.resize((32, 32), Image.ANTIALIAS)
searImg = np.asarray(searImg)
searImg = searImg / 255
searImg = torch.Tensor(searImg).unsqueeze_(dim=0)
searImg = searImg.permute((0, 3, 1, 2)).float()
sear_result = model(searImg)
sear_result = sear_result.data.numpy()
sear_result = sear_result[0]
print("sear_result=====================")
for fea in sear_result:
    print(fea)
print("sear_result=====================")

print('=====================计算mAP==============================')
# /home/zxq/code/searchImage/lable9/
ap_path = "/home/llj0571/Desktop/code/datasets/retrivalImage/avg/"
total_AP = 0
dirArr = os.listdir(ap_path)
# dirArr = dirArr[0:10]
for apfilename in dirArr:
    print(ap_path + apfilename)
    searchImg = Image.open(ap_path + apfilename)
    searchImg = searchImg.resize((32, 32), Image.ANTIALIAS)
    searchImg = np.asarray(searchImg)
    searchImg = searchImg / 255
    searchImg = torch.Tensor(searchImg).unsqueeze_(dim=0)
    searchImg = searchImg.permute((0, 3, 1, 2)).float()
    search_result = model(searchImg)
    search_result = search_result.data.numpy()
    search_result = search_result[0]  # 待检索的特征值

    resultDicList = []
    # 1、 获取数组中的所有特征值 ,计算l2 距离
    for index, featureMap in enumerate(feature_list):

        distance = 0  # 记录每个特征值的距离
        imgStr = featureMap["imgName"]
        imgArr = imgStr.split("_")
        featureArr = featureMap["feature"]
        for index2, feaValue in enumerate(featureArr):
            f2 = search_result[index2]
            dis = feaValue - f2
            distance += math.pow(dis, 2)
        l2Distance = math.sqrt(distance)
        dict2 = {"index": imgArr[1], "distance": l2Distance}
        resultDicList.append(dict2)

    # print("sort before=====================================================")
    # for featureMap in resultDicList:
    #     print(featureMap)
    # 2、 对存储map 的数组进行排序
    resultDicList = sorted(resultDicList, key=lambda k: k["distance"])
    # print("sort after=====================================================")
    # for featureMap in resultDicList:
    #     print(featureMap)

    print("计算AP======================")

    # 3、计算AP
    strArr = apfilename.split("_")
    comparelabel = strArr[1]
    # 计算AP
    p = 0
    number = 0
    for indexID, itemMap in enumerate(resultDicList):
        ind = itemMap["index"]
        # 计算P加和
        if ind == str(comparelabel):
            number += 1
            ppp = number / (indexID + 1)
            p += ppp
    # 计算AP
    ap = p / number
    total_AP += ap
    print("=======AP============" + str(ap))

mAP = total_AP / len(dirArr)
print(" mAP=======" + str(mAP))
