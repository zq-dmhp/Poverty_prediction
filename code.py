import warnings
import numpy as np
from osgeo import gdal
from tqdm import tqdm
import random
import torch
from PIL import Image
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
        datatype = gdal.GDT_Float32
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


def estimate(a, x, y, b, num):
    k = 0
    for i in range(1, num):
        for j in range(1, num):
            if x + i < len(a) and y + j < len(a[0]):
                if a[x + i, y + j] >= b:
                    k = k + 1
            if x + i < len(a) and y - j > 0:
                if a[x + i, y - j] >= b:
                    k = k + 1
            if x - i > 0 and y + j < len(a[0]):
                if a[x - i, y + j] >= b:
                    k = k + 1
            if x - i > 0 and y - j > 0:
                if a[x - i, y - j] >= b:
                    k = k + 1
    if k >= 2:
        return 0
    else:
        return 1


def estimate_2(a, x, y):
    k = 0
    if x - 2 < 0:
        x1 = 0
    else:
        x1 = x - 2
    if y - 2 < 0:
        y1 = 0
    else:
        y1 = y - 2
    if x + 3 >= len(a):
        x2 = len(a) - 1
    else:
        x2 = x + 3
    if y + 3 >= len(a[0]):
        y2 = len(a[0]) - 1
    else:
        y2 = y + 3
    if a[x, y] <= a[x1:x2, y1:y2].mean() or a[x, y] <= np.median(a[x1:x2, y1:y2]):
        return 0
    else:
        return 1


# def count(b):
#     if b < 10:
#         c = '00000' + str(b)
#     if 100 > b >= 10:
#         c = '0000' + str(b)
#     if 1000 > b >= 100:
#         c = '000' + str(b)
#     if 10000 > b >= 1000:
#         c = '00' + str(b)
#     if 100000 > b >= 10000:
#         c = '0' + str(b)
#     if b >= 100000:
#         c = str(b)
#     return c


# def find(filename):
#     kk = 0
#     kkk = 0
#     for q in filename:
#         if q == '\\':
#             kk = kk + 1
#         if kk == 6:
#             return filename[kkk + 1:-4]
#             break
#         kkk = kkk + 1
def extract_features(input, net, shape=(256, 256)):
    input = input.swapaxes(1, 0).swapaxes(1, 2)
    input_array = np.ascontiguousarray(input)
    input = Image.fromarray(input_array)
    norm = T.Compose([
        T.Resize(shape, interpolation=T.functional._interpolation_modes_from_int(3)),
        T.ToTensor(),
    ])
    # pca = joblib.load("./model/model_pca/pca_2.pkl")
    # std_x = joblib.load("./model/MinMaxScaler/MinMaxScaler_x_2.pkl")
    # 使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input = norm(input)

    # 张量重塑
    input = input.reshape(1, 3, shape[0], shape[1])

    # 将数据移到gpu上
    input = input.to(device)

    with torch.no_grad():
        net.eval()  # 评估模式, 这会关闭dropout
        output = (net(input).argmax(dim=1)).float().cpu().item()
        net.train()
        # output = net.forward(input)

    # output = output.cpu().numpy()
    # output = pd.DataFrame(output)
    # print(output.shape())

    # output_pca = pca.transform(output.values)

    # output_std = std_x.transform(output_pca)
    # print(output_std)
    return int(output)


if __name__ == '__main__':
    save_path_2 = r'E:\Poverty_prediction\Remote_sensing_image\sagment_image\7.0_2m'
    save_path_30 = r'E:\Poverty_prediction\Remote_sensing_image\sagment_image\7.0_30m'

    hangzhou = r'E:\Poverty_prediction\Remote_sensing_image\TIF\hangzhous\Level16\hang.tif'
    hangzhou_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\hangzhous\Level16\hangzhou.tif'
    hangzhou_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\hangzhous\Level16\hangzhou_w.tif'
    hangzhou_30 = r'E:\Poverty_prediction\Remote_sensing_image\杭州市(0)\Level12\杭州市(0).tif'

    ningbo = r'E:\Poverty_prediction\Remote_sensing_image\TIF\ningbos\Level16\ning.tif'
    ningbo_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\ningbos\Level16\ningbo.tif'
    ningbo_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\ningbos\Level16\ningbo_w.tif'
    ningbo_30 = r'E:\Poverty_prediction\Remote_sensing_image\宁波市(0)\Level12\宁波市(0).tif'

    wenzhou = r'E:\Poverty_prediction\Remote_sensing_image\TIF\wenzhous\Level16\wen.tif'
    wenzhou_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\wenzhous\Level16\wenzhou.tif'
    wenzhou_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\wenzhous\Level16\wenzhou_w.tif'
    wenzhou_30 = r'E:\Poverty_prediction\Remote_sensing_image\温州市(0)\Level12\温州市(0).tif'

    shaoxing = r'E:\Poverty_prediction\Remote_sensing_image\TIF\shaoxings\Level16\shao.tif'
    shaoxing_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\shaoxings\Level16\shaoxing.tif'
    shaoxing_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\shaoxings\Level16\shaoxing_w.tif'
    shaoxing_30 = r'E:\Poverty_prediction\Remote_sensing_image\绍兴市(0)\Level12\绍兴市(0).tif'

    huzhou = r'E:\Poverty_prediction\Remote_sensing_image\TIF\huzhous\Level16\hu.tif'
    huzhou_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\huzhous\Level16\huzhou.tif'
    huzhou_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\huzhous\Level16\huzhou_w.tif'
    huzhou_30 = r'E:\Poverty_prediction\Remote_sensing_image\湖州市(0)\Level12\湖州市(0).tif'

    jiaxing = r'E:\Poverty_prediction\Remote_sensing_image\TIF\jiaxings\Level16\jia.tif'
    jiaxing_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\jiaxings\Level16\jiaxing.tif'
    jiaxing_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\jiaxings\Level16\jiaxing_w.tif'
    jiaxing_30 = r'E:\Poverty_prediction\Remote_sensing_image\嘉兴市(0)\Level12\嘉兴市(0).tif'

    quzhou = r'E:\Poverty_prediction\Remote_sensing_image\TIF\quzhous\Level16\qu.tif'
    quzhou_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\quzhous\Level16\quzhou.tif'
    quzhou_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\quzhous\Level16\quzhou_w.tif'
    quzhou_30 = r'E:\Poverty_prediction\Remote_sensing_image\衢州市(0)\Level12\衢州市(0).tif'

    jinhua = r'E:\Poverty_prediction\Remote_sensing_image\TIF\jinhuas\Level16\jin.tif'
    jinhua_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\jinhuas\Level16\jinhua.tif'
    jinhua_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\jinhuas\Level16\jinhua.tif'
    jinhua_30 = r'E:\Poverty_prediction\Remote_sensing_image\金华市(0)\Level12\金华市(0).tif'

    lishui = r'E:\Poverty_prediction\Remote_sensing_image\TIF\lishuis\Level16\li.tif'
    lishui_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\lishuis\Level16\lishui.tif'
    lishui_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\lishuis\Level16\lishui_w.tif'
    lishui_30 = r'E:\Poverty_prediction\Remote_sensing_image\丽水市(0)\Level12\丽水市(0).tif'

    taizhou = r'E:\Poverty_prediction\Remote_sensing_image\TIF\taizhous\Level16\tai.tif'
    taizhou_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\taizhous\Level16\taizhou.tif'
    taizhou_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\taizhous\Level16\taizhou_w.tif'
    taizhou_30 = r'E:\Poverty_prediction\Remote_sensing_image\台州市(0)\Level12\台州市(0).tif'

    zhoushan = r'E:\Poverty_prediction\Remote_sensing_image\TIF\zhoushans\Level16\zhou.tif'
    zhoushan_n = r'E:\Poverty_prediction\Remote_sensing_image\TIF\zhoushans\Level16\zhoushan.tif'
    zhoushan_w = r'E:\Poverty_prediction\Remote_sensing_image\TIF\zhoushans\Level16\zhoushan_w.tif'
    zhoushan_30 = r'E:\Poverty_prediction\Remote_sensing_image\舟山市(0)\Level12\舟山市(0).tif'

    list_2 = [ningbo, hangzhou, wenzhou, shaoxing, huzhou, jiaxing, quzhou, jinhua, lishui, taizhou, zhoushan]
    list_1 = [ningbo_n, hangzhou_n, wenzhou_n, shaoxing_n, huzhou_n, jiaxing_n, quzhou_n, jinhua_n, lishui_n, taizhou_n,
              zhoushan_n]
    list_3 = [ningbo_w, hangzhou_w, wenzhou_w, shaoxing_w, huzhou_w, jiaxing_w, quzhou_w, jinhua_w, lishui_w, taizhou_w,
              zhoushan_w]
    list_4 = [ningbo_30, hangzhou_30, wenzhou_30, shaoxing_30, huzhou_30, jiaxing_30, quzhou_30, jinhua_30, lishui_30, taizhou_30,
              zhoushan_30]

    k = 1
    k_30 = 1
    judge = [3.14, 10.29, 20.51, 32.78, 48.11, 90, 10000]

    DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    # 网络Resnet
    finetune_net = nn.Sequential()
    finetune_net.features = torchvision.models.resnet50(pretrained=True)
    # 修改最后的全连接层,这里中的256可以改成任意数，最后的5是你的类别数
    finetune_net.output_new = nn.Sequential(nn.Dropout(0.5),
                                            nn.Linear(1000, 256),
                                            nn.ReLU(),
                                            nn.Linear(256, 6))
    net = finetune_net.to(DEVICE)
    # 加载预训练模型
    net.load_state_dict(torch.load(r'E:\Poverty_prediction\model\pre_training_model\ResNet50_7.pth'))
    for i in range(7):
        os.mkdir(save_path_2 + '\\' + str(i))
        os.mkdir(save_path_30 + '\\' + str(i))
    for p in range(0, 11):
        data_1, y_1, x_1 = readTif(list_1[p])
        data_2, y_2, x_2 = readTif(list_2[p])
        data_3, y_3, x_3 = readTif(list_3[p])
        data_4, y_4, x_4 = readTif(list_4[p])
        t_1 = int(x_2 / x_1)
        t_2 = int(y_2 / y_1)
        t_11 = int(x_3 / x_1)
        t_22 = int(y_3 / y_1)
        t_111 = int(x_4 / x_1)
        t_222 = int(y_4 / y_1)
        for i in tqdm(range(0, x_1)):
            for j in range(0, y_1):
                k_1 = data_1[i, j]
                k_2 = data_2[:, i * t_1:(i + 1) * t_1, j * t_2:(j + 1) * t_2]
                #print(k_2.shape)
                k_3 = data_3[i * t_11:(i + 1) * t_11, j * t_22:(j + 1) * t_22]
                k_4 = data_4[:, i * t_111:(i + 1) * t_111, j * t_222:(j + 1) * t_222]
                #print(k_4.shape)
                if k_1 > 0 and np.min(k_2) != 0:
                    hor = random.choice(
                        [True, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False,
                         False, False, False, False, False, False, False, False, False, False, False, False, False])

                    if k_1 <= judge[0] and estimate(data_1, i, j, judge[0], 7) and hor:
                        writeTiff(save_path_2 + '\\0\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                        writeTiff(save_path_30 + '\\0\\' + str(k) + "_" + str(k_1) + ".tif", k_4)
                        k = k + 1
                    elif (not k_3.__contains__(41)) and (not k_3.__contains__(42)):
                        hor_2 = random.choice([True, False, False])
                        hor_3 = random.choice([True, False, False])
                        hor_4 = random.choice([True, False])
                        if judge[0] <= k_1 <= judge[1] and estimate(data_1, i, j, judge[1], 5) \
                                and estimate_2(data_1, i, j) and extract_features(k_2,net, shape=(256, 256)) == 1:
                            writeTiff(save_path_2 + '\\1\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            writeTiff(save_path_30 + '\\1\\' + str(k) + "_" + str(k_1) + ".tif", k_4)
                            k = k + 1
                        elif judge[1] <= k_1 <= judge[2] and estimate(data_1, i, j, judge[2], 2) \
                                and estimate_2(data_1, i,j) and extract_features(k_2,net, shape=(256, 256)) == 2:
                            writeTiff(save_path_2 + '\\2\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            writeTiff(save_path_30 + '\\2\\' + str(k) + "_" + str(k_1) + ".tif", k_4)
                            k = k + 1
                        elif judge[2] <= k_1 <= judge[3] and estimate_2(data_1, i, j) and extract_features(k_2,net, shape=(256, 256)) == 3:
                            writeTiff(save_path_2 + '\\3\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            writeTiff(save_path_30 + '\\3\\' + str(k) + "_" + str(k_1) + ".tif", k_4)
                            k = k + 1
                        elif judge[3] <= k_1 <= judge[4] and estimate_2(data_1, i, j) and extract_features(k_2,net, shape=(256, 256)) == 4:
                            writeTiff(save_path_2 + '\\4\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            writeTiff(save_path_30 + '\\4\\' + str(k) + "_" + str(k_1) + ".tif", k_4)
                            k = k + 1
                        elif judge[4] <= k_1 <= judge[5] and estimate_2(data_1, i, j):
                            writeTiff(save_path_2 + '\\5\\' + str(k) + "_" + str(k_1) + ".tif", k_2)
                            writeTiff(save_path_30 + '\\5\\' + str(k) + "_" + str(k_1) + ".tif", k_4)
                            k = k + 1

    # for i in tqdm(range(0, 135, 3)):
    #     for j in range(0, 157, 3):
    #         k_1 = np.mean(data_1[i:i + 3, j:j + 3])
    #         ii = i * 3 * 208
    #         jj = j * 3 * 208
    #         k_2 = data_2[:, ii:ii + 3 * 208, jj:jj + 3 * 208]
    #         if np.sum(k_2) != 0:
    #             writeTiff(r"D:\arcdata\xs_" + count(k) + "_" + str(k_1) + ".tif", k_2)
    #             k = k + 1
