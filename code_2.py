import warnings
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import random
import torch
from PIL import Image
import cv2 as cv

warnings.filterwarnings("ignore")
# import segmentation_models_pytorch as smp
# import torch
# from utils import convert_from_color_segmentation
import os
from torchvision import transforms as T

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import torch.nn as nn
import torchvision.models


# 读取tif数据集
def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset is None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  获取数据
    if data_width == 0 and data_height == 0:
        data_width = width
        data_height = height
    print("width:", width, "height:", height)
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)
    print("fileName     finish!")
    return data, width, height


# 保存tif文件函数
def writeTiff(fileName, data, im_geotrans=(0, 0, 0, 0, 0, 0), im_proj=""):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_UInt16
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(im_height), int(im_bands), datatype)
    if dataset is not None:
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset


if __name__ == '__main__':
    save_path = r"E:\Zhejiang_Landsat5_8\2020\code2"
    night_path = r"E:\Zhejiang_Landsat5_8\2020\VNL_2020\VNL_zhejiang"
    day_path = r"E:\Zhejiang_Landsat5_8\2020\Reflectance_2020_Composite\zhejiang"
    judge = [3.14, 10.29, 20.51, 32.78, 48.11, 90, 10000]
    k = 1
    for i in range(6):
        os.mkdir(save_path + '\\' + str(i))
    for p in range(2, 7):
        if p == 3:
            continue
        data_1, y_1, x_1 = readTif(night_path + str(p) + "_mask.tif")
        data_2, y_2, x_2 = readTif(day_path + str(p) + "_mask.tif")
        t_1 = int(x_2 / x_1)
        t_2 = int(y_2 / y_1)
        for i in range(0, x_1, 3):
            for j in range(0, y_1, 3):
                k_1 = np.mean(data_1[i:i+3, j:j+3])
                k_2 = data_2[:, i * t_1:(i + 3) * t_1, j * t_2:(j + 3) * t_2]
                if k_1 > 0 and np.min(k_2) != 0:
                    hor = random.choice(
                        [True, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False])
                    if k_1 <= judge[0] and hor:
                        k_2 = (k_2*256).astype(int)
                        writeTiff(save_path + '\\0\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                        k = k + 1
                    else:
                        # hor_2 = random.choice([True, False, False, False, False, False, False, False, False, False])
                        # hor_3 = random.choice([True, False, False, False, False])
                        # hor_4 = random.choice([True, False])
                        if judge[0] <= k_1 <= judge[1]:
                            k_2 = (k_2*256).astype(int)
                            writeTiff(save_path + '\\1\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            k = k + 1
                        elif judge[1] <= k_1 <= judge[2]:
                            k_2 = (k_2*256).astype(int)
                            writeTiff(save_path + '\\2\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            k = k + 1
                        elif judge[2] <= k_1 <= judge[3]:
                            k_2 = (k_2*256).astype(int)
                            writeTiff(save_path + '\\3\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            k = k + 1
                        elif judge[3] <= k_1 <= judge[4]:
                            k_2 = (k_2*256).astype(int)
                            writeTiff(save_path + '\\4\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            k = k + 1
                        elif judge[4] <= k_1 <= judge[5]:
                            k_2 = (k_2*256).astype(int)
                            writeTiff(save_path + '\\5\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            k = k + 1

