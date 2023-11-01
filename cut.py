import warnings
import numpy as np
from osgeo import gdal
warnings.filterwarnings("ignore")
import os
import cv2 as cv
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

def saveimage(img, img_name, img_path):
    """用于保存处理后的文件"""
    cv.imwrite(os.path.join(img_path, img_name), img)
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


def cutimage(image, x0, y0, x1, y1):
    """用于获取根据左上角,与右下角的坐标所截取的图像"""
    return image[:,y0:y1, x0:x1]  # , label[y0:y1, x0:x1]

if __name__ == '__main__':
    path = r'E:\Poverty_prediction\Remote_sensing_image\2000\2000.tif'
    #path = r'E:\Poverty_prediction\Remote_sensing_image\Raw_NTL_image\ningbo\1_12.5697155.tif'
    save_path = r'E:\DASR-main\experiment\blindsr_x4_bicubic_iso\results\2000\remote'
    data, width, height = readTif(path)
    #data = cv.imread(path,-1)
    h_num, w_num = 40,40
    input_size = (width//h_num, height//w_num)
    print(input_size)
    in_w, in_h = input_size[0], input_size[1]
    print(in_h ,in_w)
    count = 0
    for i in range(0, h_num):
        for j in range(0, w_num):
            count += 1
            print(count)
            #print(j * in_w, i * in_h, j * in_w + in_w, i * in_h + in_h)
            cutImage = cutimage(data, j * in_w, i * in_h, j * in_w + in_w, i * in_h + in_h)
            name = str(count) + '.tif'
            # dataIncrease.saveimage(cutImage, name, cutLabel, name)
            # cnt_array = np.where(cutImage, 0, 1)
            # print(int(np.sum(cnt_array)))
            # if int(np.sum(cnt_array)) != int(in_h*in_w*3):
            #print(cutImage)
            writeTiff(save_path + '\\' + name, cutImage)


            #saveimage(cutImage, name, save_path)

