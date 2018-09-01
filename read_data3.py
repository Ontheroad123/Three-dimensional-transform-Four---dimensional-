#!/usr/bin/env python
# -*- coding:utf-8 -*-

from PIL import Image, ImageFilter
import numpy as np
import os
from read_img import endwith

class GaussianBlur(ImageFilter.Filter):
    def __init__(self, radius=2, bounds=None):
        self.radius = radius
        self.bounds = bounds

    def filter(self, image):
        if self.bounds:

            clips = image.crop(self.bounds).gaussian_blur(self.radius)
            image.paste(clips, self.bounds)
            return image
        else:
            return image.gaussian_blur(self.radius)

def read_file(path):
    img_list = []
    label_list = []
    dir_counter = 0

    # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
        child_path = os.path.join(path, child_dir)

        for dir_image in os.listdir(child_path):

            if endwith(dir_image, 'jpg'):
                #img = cv2.imread(os.path.join(child_path, dir_image))
                img = Image.open(os.path.join(child_path, dir_image))
                #图片归一化处理
                #img = cv2.resize(img, (255, 255), interpolation=cv2.INTER_CUBIC)
                img=img.resize((255,255))
                arr = np.asarray(img, dtype="float32")
                #高斯模糊处理
                image1 = img.filter(GaussianBlur(radius=1))
                arr1 = np.asarray(image1, dtype="float32")
                image2 = img.filter(GaussianBlur(radius=3))
                arr2 = np.asarray(image2, dtype="float32")
                image3 = img.filter(GaussianBlur(radius=5))
                arr3 = np.asarray(image3, dtype="float32")
                #合成四维矩阵
                new = np.empty((255, 255, 3, 4), dtype="float32")
                new[:, :, :, 0] = arr
                new[:, :, :, 1] = arr1
                new[:, :, :, 2] = arr2
                new[:, :, :, 3] = arr3

                img_list.append(new)
                label_list.append(dir_counter)

        dir_counter += 1

    # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)

    return img_list, label_list, dir_counter

if __name__ == '__main__':
    path="/home/hq/desktop/cat_dog/c-d-data/train1/"
    read_file(path)











