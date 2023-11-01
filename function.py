import pandas as pd
import random
import numpy as np
import os

def DataAugmentation(image, label, mode):
    if(mode == "train"):
        hor = random.choice([True, False])
        if(hor):
            #  图像水平翻转
            image = np.flip(image, axis = 1)
            #label = np.flip(label, axis = 1)
        ver = random.choice([True, False])
        if(ver):
            #  图像垂直翻转
            image = np.flip(image, axis = 0)
            #label = np.flip(label, axis = 0)
        #stretch = random.choice([True, False])
        #if(stretch):
            #image = truncated_linear_stretch(image, 0.5)

    if(mode == "val"):
        stretch = random.choice([0.8, 1, 2])
        # if(stretch == 'yes'):
        # 0.5%线性拉伸
        #image = truncated_linear_stretch(image, stretch)
    return image, label


def Image_gounp(labels_list):
    images_list = []
    for i in range(len(labels_list)):
        label_list = [os.path.join(labels_list[i], item) for item in os.listdir(labels_list[i])]
        images_list.extend(label_list)
    return images_list

