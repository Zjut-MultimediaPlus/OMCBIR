
def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(20) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomColor(root_path, img_name): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    image = Image.open(os.path.join(root_path, img_name))
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(10, 21) / 10.  # 随机因子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(0, 31) / 10.  # 随机因子
    return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted

def brightnessEnhancement(root_path,img_name):#亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened

def colorEnhancement(root_path,img_name):#颜色增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_col = ImageEnhance.Color(image)
    color = 1.5
    image_colored = enh_col.enhance(color)
    return image_colored

from PIL import Image
from PIL import ImageEnhance
import os
# import cv2
import numpy as np
imageDir="/Users/qing/Desktop/7datasets/hkwy/totalData/A"     #要改变的图片的路径文件夹
saveDir="/Users/qing/Desktop/7datasets/hkwy/totalData/A_aug"   #要保存的图片的路径文件夹

for name in os.listdir(imageDir):

   # 原始图像
    saveName= name[:-4]+"id.jpg"
    image = Image.open(os.path.join(imageDir, name))
    image.save(os.path.join(saveDir,saveName))

    # 亮度增强
    saveName= name[:-4]+"be.jpg"
    saveImage=brightnessEnhancement(imageDir,name)
    saveImage.save(os.path.join(saveDir,saveName))

   # 翻转图像
    saveName= name[:-4]+"fl.jpg"
    saveImage=flip(imageDir,name)
    saveImage.save(os.path.join(saveDir,saveName))
   # 旋转图像
    saveName= name[:-4]+"ro.jpg"
    saveImage=rotation(imageDir,name)
    saveImage.save(os.path.join(saveDir,saveName))

    # 对比度增强
    saveName = name[:-4] + "ce.jpg"
    saveImage = contrastEnhancement(imageDir, name)
    saveImage.save(os.path.join(saveDir, saveName))


import cv2
import numpy as np
file = 'data_batch_1'
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo,encoding='latin1')
    return dict
dict1 = unpickle(file)
for i in range(1000):#我只保存的100张
    img = dict1["data"][i]#得到图片的数据
    img = np.reshape(img, (3, 32,32))  #转为三维图片数组
    img = img.transpose((1,2,0))#通道转换为CV2的要求形式
    img_name = dict1["filenames"][i]#拿到图片的名字
    img_label = str(dict1["labels"][i])#拿到图片的标签
    cv2.imwrite("./p/"+img_label+"_"+img_name,img)#保存
