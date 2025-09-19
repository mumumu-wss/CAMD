#coding:utf-8
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import torchvision.transforms as transforms
import argparse
import torch
# import matplotlib.pyplot as plt
def get_resized_label(img_size, resized_size, label):

    #leftx, lefty, width, hight
    label_copy = label.copy()
    label_copy[::2] = label_copy[::2] * 1.0 * resized_size[0] / img_size[0]
    label_copy[1::2] = label_copy[1::2] * 1.0 * resized_size[1] / img_size[1]
    return label_copy
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def make_img(part_dir, partition):
    img = []
    with open(part_dir) as f:
        lines = f.readlines()
        for line in lines:
            pic_dir, num = line.split()
            if num == partition: 
                img.append(pic_dir)
    return img
def get_imageNameDict(path_dir="/data/Leimingxuan/data/LFWA/lfw/"):
    dirs = os.listdir(path_dir)
    dict={}
    for i in range(len(dirs)):
        dict[dirs[i]]=i
    return dict

Attr_Threshold_InFace = torch.Tensor([0.5 for i in range(40)])

class CelebA(data.Dataset):
    def __init__(self, part_dir, attr_dir, partition, img_dir, transform):
        self.attr = np.zeros((202600, 40))
        with open(attr_dir) as f:
            f.readline()
            f.readline()
            lines = f.readlines()
            id = 0
            for line in lines:
                vals = line.split()
                id += 1
                for j in range(40):
                    self.attr[id, j] = int(vals[j+1])
        self.img= make_img(part_dir, partition)
        self.length = len(self.img)
        self.transform = transform
        self.img_dir = img_dir

        #只要用到训练集的fll标注即可，这里为了存取方便，把fll初始值赋为-1,表示图片中没有识别人脸
        self.fll = np.ones((202600, 144))*(-1)
        path = os.path.dirname(__file__)
        # print(path)
        with open(path+'/landmarks.txt') as f2:
            lines = f2.readlines()
            for line in lines:
                vals = line.split()
                img_path = vals[0]
                # print img_path
                img_num = int(img_path.split('/')[-1].split('.')[0])              #看文件具体放在哪里的层数可能需要改变这个数值
                for j in range(144):
                    self.fll[img_num, j] = int(vals[j+1])





    def __getitem__(self, index):



        # 画出变化后的bounding box观测是否正确
        # figure(2)
        # draw_fll(image, self.fll[index, :])
        # imshow(image)
        #
        # figure(3)
        # draw_bounding_box(image, self.box[index, :])
        # imshow(image)
        # show()
        if index >= len(self.imgs) or index < 0:
            print(f"Index out of range: {index}, length of imgs: {len(self.imgs)}")
        image = pil_loader(os.path.join(self.img_dir, self.img[index]))
        # print image
        w, h = image.size
        # m = min(w, h)
        new_w = int(1.0 * w / h * 224)
        image = transforms.Resize((224, new_w))(image)
        image = np.array(image)

        image_new = np.zeros((224, 224, 3))
        add_num = (224 - new_w)/2
        image_new[:, add_num:add_num+new_w, :] = image.copy()        #如果是其他图片的大小
        image_new = Image.fromarray(np.uint8(image_new))


        # image_new.show()


        # plt.imshow(image_new)
        # plt.show()
        # print image.shape
        if self.transform is not None:
            image_new = self.transform(image_new)    #这里的输入好像必须是image类型
        id = int(self.img[index].split('.')[0])

        #resize操作改变的fll标签
        fll = get_resized_label((w, h), (1, 1), self.fll[id, :])  #函数里边是copy数组,避免改变原来的self,fll下个epoch有错误
        # print fll
        #在两侧添加像素改变的fll标签, 由于celebA 218*178因此只有横坐标的标签需要变化
        fll[::2] = 1.0*add_num/224 + fll[::2] / 0.5 * (0.5 - 1.0*add_num/224)
        # print fll

        # print("id={}".format(id))
        # print("index ={}".format(index)) 训练集的时候两者相差１测试集的时候两者
        # print index , type(self.attr[id,:])
        return image_new, self.attr[id, :], fll


    def __len__(self):
        return self.length


class LFWA(data.Dataset):
    def __init__(self, attr_dir, mode, img_dir, transform ,few_shot_ratio=1):
        #self.image是一个字符数组 每个元素形如 Miyako_Miyazaki/Miyako_Miyazaki_0001.jpg
        # 参数包括 attr_dir 属性标签文件
        # mode： train 和 test
        if mode == 'train':
            #训练集把funnel和原始的图片都加入作为训练
            # self.attr = np.zeros((6572*2, 40))
            self.attr = np.zeros((6572, 40))
        else:
            self.attr = np.zeros((6571, 40))

        with open(attr_dir) as f:
            f.readline()
            f.readline()
            lines = f.readlines()

            if mode =='train':
                flag = 0
            else:
                flag = 1

            self.imgs = []
            idx = 0

            for line in lines:
                flag = 1 - flag
                if flag % 2 == 0:
                    continue
                # 每一行有可能有76-79个，其中前2个或者3个是人名，后边依次是每个人图片的数量和73个属性

                #把图像的路径加入到self.imgs

                vals = line.split()
                len_name = len(vals) - 74 #73个属性加上 1个num(这个人的第几张图片)

                person_name = vals[0]
                for i in range(1, len_name):
                    person_name += '_' + vals[i]
                img_name = person_name + '/' + person_name + '_' + str(vals[len_name]).zfill(4)+'.jpg'

                self.imgs.append(img_name)
                # #把剪切过和未剪切的图片都计入训练集防止过拟合
                # if mode == 'train':
                #     self.imgs.append("/data/Leimingxuan/data/LFWA/lfw/" + img_name)

                #把图像的标签加入到self.attr
                index = [65, 36, 56, 60, 13, 30 ,41, 39, 10, 11,
                         21, 12, 35, 20, 49, 15, 47, 59, 61, 69,
                         1,  43, 17, 37, 46, 51, 64, 40, 29, 62,
                         31, 18, 28, 27, 71, 50, 67, 73, 72, 7]

                #如果是列表好像不支持temp[temp>0] = 1 的操作
                # temp = np.array([(1 / (1 + np.exp(-float(s)))) for s in vals[-74:]])
                temp = np.array([float(s) for s in vals[-74:]])   # temp 形如 [num, 0.001, -0.221, ...]

                temp[temp >= 0] = 1
                temp[temp < 0] = 0                     # temp 形如[num=1, -1, 1, -1...]
                # print temp
                self.attr[idx, :] = [temp[i] for i in index]

                if temp[57] > 0:
                    self.attr[idx, 2] = 1

                idx += 1

        path = os.path.dirname(__file__)
        if mode == 'train':
            flag = 0
        else:
            flag = 1

        # few-shot
        if mode == 'train':
            if few_shot_ratio < 1:
                few_shot_step = int(1/few_shot_ratio)
                self.imgs = self.imgs[::few_shot_step]
                self.attr = self.attr[::few_shot_step]
        self.length = len(self.imgs)
        self.transform = transform
        self.img_dir = img_dir
        # print self.imgs[1]

    def __getitem__(self, index):
        image = pil_loader(os.path.join(self.img_dir, self.imgs[index]))

        w, h = image.size
        # m = min(w, h)
        new_w = int(1.0 * w / h * 224)
        image = transforms.Resize((224, new_w))(image)
        image = np.array(image)

        image_new = np.zeros((224, 224, 3))
        add_num = (224 - new_w) / 2

        list1 = image_new[:,int(add_num):int(add_num+new_w), :]
        image_new = np.resize(image,(list1.shape[0],list1.shape[1],list1.shape[2]))

        image_new = Image.fromarray(np.uint8(image_new))
        image_new = image_new.convert('RGB')

        if self.transform is not None:
            image_new = self.transform(image_new)  # 这里的输入好像必须是image类型

        return image_new, self.attr[index, :]


    def __len__(self):
        return self.length

if __name__ == '__main__':
    import os
    dirs=get_imageNameDict()
    print('done??')
