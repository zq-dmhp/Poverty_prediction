from Resnet_50 import *
from osgeo import gdal
import numpy as np
import datetime
import math
import warnings

warnings.filterwarnings("ignore")
import sys
# import segmentation_models_pytorch as smp
import torch
import cv2
from torchvision import transforms as T
# from utils import convert_from_color_segmentation
from tqdm import tqdm
import albumentations as alb
import joblib
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
import pandas as pd
from train import get_net
import cv2 as cv
from sklearn.preprocessing import MinMaxScaler


# 读取tif数据集
def readTif(fileName, xoff=0, yoff=0, data_width=0, data_height=0):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    #  栅格矩阵的列数
    width = dataset.RasterXSize
    #  栅格矩阵的行数
    height = dataset.RasterYSize
    #  获取数据
    if (data_width == 0 and data_height == 0):
        data_width = width
        data_height = height
    data = dataset.ReadAsArray(xoff, yoff, data_width, data_height)

    return data


# 保存tif文件函数
def writeTiff(fileName, data, im_geotrans=(0, 0, 0, 0, 0, 0), im_proj=""):
    if 'int8' in data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(data.shape) == 3:
        im_bands, im_height, im_width = data.shape
    elif len(data.shape) == 2:
        data = np.array([data])
        im_bands, im_height, im_width = data.shape

    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(fileName, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(data[i])
    del dataset


def TifCroppingArray(img):
    #  裁剪链表
    TifArrayReturn = []
    #  列上图像块数目
    ColumnNum = int(img.shape[0] / each_size)
    #  行上图像块数目
    RowNum = int(img.shape[1] / each_size)
    for i in range(ColumnNum):
        TifArray = []
        for j in range(RowNum):
            cropped = img[i * each_size: i * each_size + each_size,
                      j * each_size: j * each_size + each_size]
            TifArray.append(cropped)
        TifArrayReturn.append(TifArray)
        # print(TifArrayReturn)

    return TifArrayReturn


def Result(shape, TifArray, npyfile):
    # shape = (shape[0] // each_size,shape[1] // each_size)
    result = np.zeros(shape, np.uint8)
    for i in range(len(TifArray)):
        # result[i:i+each_size,:] = npyfile[i]
        for j in range(len(TifArray[0])):
            result[i * each_size: i * each_size + each_size, j * each_size: j * each_size + each_size] = npyfile[i][j]
    return result


def extract_features(input, net, shape):
    norm = T.Compose([
        T.ToTensor(),
    ])
    pca = joblib.load("./model/model_pca/pca_500m_1.pkl")
    std_x = joblib.load("./model/MinMaxScaler/MinMaxScaler_x_500m_1.pkl")
    # 使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #print(input.shape)
    #可删减的
    #input = input.astype(np.float64)
    #
    #print(input.reshape)
    input = norm(input)
    # 张量重塑
    input = input.reshape(1, 3, shape[0], shape[1])
    # 将数据移到gpu上
    input = input.to(device)

    with torch.no_grad():
        # output = net.forward(input)
        net.eval()  # 评估模式, 这会关闭dropout
        output = (net(input)).float()
        net.train()

    output = output.cpu().numpy()
    output = pd.DataFrame(output)
    # print(output.shape())
    # print(output)

    output_pca = pca.transform(output.values)

    output_std = std_x.transform(output_pca)
    # print(output_std)
    return output_std


if __name__ == '__main__':

    #each_size = 256 #2m
    each_size = 96 #10m
    area_perc = 0  # 0.5
    in_channels = 3
    # model_name = "deeplabv3+"
    # encoder_name = "resnet101"
    # TifPath = r"./datasets/paddy/test/114028_20140921.TIF"
    # TifPath = r"./datasets/paddy/114028_20140921.TIF"
    # TifPath = r'E:\Poverty_prediction\Remote_sensing_image\2005\1.1\LT51180392005331BJC00_Com1.tif'#这里改
    TifPath = r'E:\Poverty_prediction\Remote_sensing_image\2000\hangzhou2023-10m.tif'

    RepetitiveLength = int((1 - math.sqrt(area_perc)) * each_size / 2)

    feature_rank = np.load('./model/feature_rank/feature_rank_500m_1.npy')
    feature_rank = feature_rank.tolist()

    Result_Image = r"杭州_2023_96.tif"  # 还有这
    ResultPath = r"./result/" + Result_Image
    TiffResultPath = r"./result/Tif/2000/" + Result_Image

    # big_image = readTif(TifPath)
    # big_image = big_image.swapaxes(1, 0).swapaxes(1, 2)

    # big_image = cv2.imread(TifPath, cv2.IMREAD_UNCHANGED)
    data = gdal.Open(TifPath)
    width = data.RasterXSize
    height = data.RasterYSize
    proj = data.GetProjection()
    geotrans = data.GetGeoTransform()
    print("width:", width, "height:", height)
    data = data.ReadAsArray(0, 0, width, height)
    # data = data[1:]

    big_image = data
    print(data.shape)
    big_image = big_image.swapaxes(1, 0).swapaxes(1, 2)
    print(big_image.shape)

    TifArray = TifCroppingArray(big_image)

    # 将模型加载到指定设备DEVICE上
    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 网络Resnet50，NTL_label分为3类
    net = get_net(DEVICE)
    # 加载预训练模型
    net.load_state_dict(torch.load(r'E:\Poverty_prediction\model\pre_training_model\ResNet50_7.pth'))
    # 删除最后的全连接层
    classifier = nn.Sequential()
    net.output_new = classifier

    # stacking_model = joblib.load("./model/model_stack/stack_2.pkl")
    RidgeCV_model = joblib.load("./model/model_ridge/ridge_500m_1.pkl")

    std_y = joblib.load("./model/MinMaxScaler/MinMaxScaler_y_500m_1.pkl")
    # print(std_y.data_max_)
    # print(std_y.data_min_)
    # pred = std_y.inverse_transform([pred])
    std_y_max = std_y.data_max_
    print("Model Runing on the {}".format(DEVICE))

    print("The Length of TifArray is {}".format(len(TifArray)))
    print("The Length of TifArray is {}".format(len(TifArray[0])))
    # batch_size = 30
    predicts = []
    for i in tqdm(range(len(TifArray))):
        img_list = []
        for j in range(len(TifArray[0])):
            image = TifArray[i][j]
            cnt_array = np.where(image, 0, 1)
            if int(np.sum(cnt_array)) <= int(256 * 256):
                image = extract_features(image, net, shape=(each_size, each_size))
                image = image[:, feature_rank[:20]]

                # image = torch.tensor(image)
                # image = image.permute(2, 0, 1)
                # image = image.cuda()[None]
                # image = image.type(torch.cuda.FloatTensor)
                # pred = np.zeros((1, 1, each_size, each_size))
                # 2000 0.33 2020 0.53
                pred = RidgeCV_model.predict(image)
                # print(pred)
                # std_y = joblib.load("./model/MinMaxScaler/MinMaxScaler_y_500m_1.pkl")
                # pred = std_y.inverse_transform([pred])
                # print(pred.shape)
            else:
                pred = [0]

                # pred = pred.astype(np.uint8)
                # print(pred[0])
                # print(pred)
            if pred[0] < 0:
                pred = [0]
            0

            img_list.append(pred[0] * 180)
        predicts.append(img_list)

    print("Finshed !")

    # 保存结果predictspredicts
    result_shape = (big_image.shape[0], big_image.shape[1])
    result_data = Result(result_shape, TifArray, predicts)
    cv.imwrite(r'E:\Poverty_prediction\result\Tif\2000\2000_4_test_8.png', result_data)
    print(result_data.shape)

    datatype = gdal.GDT_Byte
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(TiffResultPath, int(width), int(height), int(1), datatype)
    dataset.SetGeoTransform(geotrans)  # 写入仿射变换参数
    dataset.SetProjection(proj)  # 写入投影
    dataset.GetRasterBand(1).WriteArray(result_data)

    # img = convert_from_color_segmentation(result_data)
    # arr, num = np.unique(result_data[:, :, 2], return_counts=True)
    #
    # print(arr)
    # print(num)
    # cv2.imwrite(ResultPath, result_data)
# writeTiff(ResultPath,  )
