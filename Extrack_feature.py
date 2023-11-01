from Resnet_50 import *
from MyDataset import *
import xlwt
import numpy as np
import pandas as pd
from train import get_net
#PCA降维

def PCA_svd(X, k, center=True): #出特征X和我们想要得到的维度k，输出就是降维后的特征：
  n = X.size()[0]
  ones = torch.ones(n).view([n,1])
  h = ((1/n) * torch.mm(ones, ones.t())) if center  else torch.zeros(n*n).view([n,n])
  H = torch.eye(n) - h
  H = H.cuda()
  X_center =  torch.mm(H.double(), X.double())
  u, s, v = torch.svd(X_center)
  components  = v[:k].t()
  #explained_variance = torch.mul(s[:k], s[:k])/(n-1)
  return components

def save(data, path):
  data_df = pd.DataFrame(data)
  data_df.to_csv(path, index=False)

def cutimage(image, x0, y0, x1, y1):
    """用于获取根据左上角,与右下角的坐标所截取的图像"""
    return image[y0:y1, x0:x1]  # , label[y0:y1, x0:x1]


if __name__ == '__main__':
    Povertry_image_path = r'E:\Poverty_prediction\Dataset\Images_2'
    #Povertry_image_path = r'D:\Poverty_prediction\Raw_Dataset\Remote_sensing_image\images_val'
    Povertry_label_path = r'E:\Poverty_prediction\Dataset\Poverty_data.csv'
    #Povertry_label_path = r'D:\Poverty_prediction\Raw_Dataset\Poverty_data.csv'
    #shape = (512,512)
    shape = (256, 256)
    norm = transforms.Compose([
        transforms.Resize(shape, interpolation=f._interpolation_modes_from_int(3)),
        # BICUBIC要调整大小，请对所有可能影响输出值的像素使用三次插值法计算输出像素值.对于其他变换，在输入图像中使用4x4环境的三次插值
        transforms.ToTensor(),
    ])
    Povertry_label = pd.read_csv(Povertry_label_path,encoding='gbk')
    #print(Povertry_label)
    # 使用GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 网络Resnet50，NTL_label分为3类
    net = get_net(device)
    # 加载预训练模型
    net.load_state_dict(torch.load(r'E:\Poverty_prediction\model\pre_training_model\ResNet50_7.pth'))
    #删除最后的全连接层
    classifier = nn.Sequential()
    net.output_new = classifier
    # net = ResNet50(num_classes=3).to(device)
    # # 加载预训练模型
    # net.load_state_dict(torch.load(r'D:\Poverty_prediction\model\ResNet50_50.pth'))
    # 删除最后的全连接层
    # classifier = nn.Sequential()
    # net.fc = classifier
    print("----------------start-----------------")
    images_list = [os.path.join(Povertry_image_path, item) for item in os.listdir(Povertry_image_path)]
    print(images_list)
    num_image = len(images_list)
    total_num = 0
    bimage = torch.tensor([]).to(device)
    label_list = torch.tensor([]).to(device)
    for item in range(0, num_image):
        simage = torch.tensor([]).to(device)
        train_image = cv.imread(images_list[item])
        ID = images_list[item].split('_')[-4]
        ID = ID.split('\\')[-1]
        #print(ID)
        label = Povertry_label.loc[Povertry_label['ID']==int(ID)]['total_capita']
        print(label)
        label = np.array(label)
        label = torch.tensor([label]).to(device)
        label_list = torch.cat((label_list, label), 0)
        # 处理image
        h, w, _ = train_image.shape
        h_num, w_num = (5, 5)  # train_label.shape
        in_h, in_w = h // h_num, w // w_num
        count = 0
        #img_name = os.listdir(img_root_path)[item]
        for i in range(0, h_num):
            for j in range(0, w_num):
                total_num += 1
                # cutImage, cutLabel = dataIncrease.cutimage(train_image, train_label,j*in_w, i*in_h, j*in_w+in_w, i*in_h+in_h)
                cutImage = cutimage(train_image,
                                    j * in_w, i * in_h, j * in_w + in_w, i * in_h + in_h)
                image_array = np.ascontiguousarray(cutImage)
                PIL_image = Image.fromarray(image_array)  # 这里ndarray_image为原来的numpy数组类型的输入
                image_norm = norm(PIL_image)
                # 张量重塑
                #image_norm_re = image_norm.reshape(1, 3, 512, 512)
                image_norm_re = image_norm.reshape(1, 3, 256, 256)
                # 将数据移到gpu上
                image_norm_re = image_norm_re.to(device)
                with torch.no_grad():
                    #output = net.forward(image_norm_re)
                    net.eval()  # 评估模式, 这会关闭dropout
                    output = (net(image_norm_re)).float()
                    net.train()
                #print(output)
                simage = torch.cat((simage,output),0)

                count += 1
                #print(count)
        #simage_PCA = PCA_svd(simage , 10)
        #print(simage_PCA.shape)
        simage_mean = torch.mean(simage, axis=0)
        #simage_mean = torch.mean(simage, axis=0)
        #print(simage_mean.shape)
        simage_mean = simage_mean.reshape(1, 1000)
        #print(simage_mean)
        print(label_list.shape)
        bimage = torch.cat((bimage, simage_mean), 0)
        print(bimage.shape)
        #bimage = bimage.cpu().numpy()
        #print(bimage.bimage)
        #print(bimage.cpu().numpy())
        save(bimage.cpu().numpy(),r"E:\Poverty_prediction\Dataset\train_data\train_1000_X_500m_2.csv")
        save(label_list.cpu().numpy(), r"E:\Poverty_prediction\Dataset\train_data\train_1000_Y_500m_2.csv")
        #torch.save(bimage, r"D:\Poverty_prediction\Dataset\train_data\X-2048-3-file")
        #torch.save(label_list, r"D:\Poverty_prediction\Dataset\train_data\Y-2048-3-file")
    #保存张量
    print(bimage.cpu().numpy().dtype())
    save(bimage.cpu().numpy(), r"E:\Poverty_prediction\Dataset\train_data\train_1000_X_500m_2.csv")
    save(label_list.cpu().numpy(), r"E:\Poverty_prediction\Dataset\train_data\train_1000_Y_500m_2.csv")
    #torch.save(bimage, r"D:\Poverty_prediction\Dataset\train_data\X-2048-3-file")
    #torch.save(label_list, r"D:\Poverty_prediction\Dataset\train_data\Y-2048-3-file")
    print('total:', total_num)
    print("----------------end-----------------")






