# # -*- coding: utf-8 -*-
# ##scipy==1.2.1

# import h5py
# import numpy as np
# import scipy.io as sio
# import scipy.misc as sc
# import glob
# import re
# import math
# # Parameters
# height = 512 # Enter the image size of the model.
# width  = 512 # Enter the image size of the model.
# channels = 3 # Number of image channels

# train_number = 1006 # Randomly assign the number of images for generating the training set.
# val_number = 125   # Randomly assign the number of images for generating the validation set.
# test_number = 127  # Randomly assign the number of images for generating the test set.
# all = int(train_number) + int(val_number) + int(test_number)

# ############################################################# Prepare your data set #################################################
# Tr_list = glob.glob("/data/fengxiao/UltraLight-VM-UNet/data/AD/image"+'/*.png')   # Images storage folder. The image type should be 24-bit png format.





# Data_train_2018    = np.zeros([all, height, width, channels])
# Label_train_2018   = np.zeros([all, height, width])

# print('Reading')
# print(len(Tr_list))
# for idx in range(len(Tr_list)):
#     print(idx+1)
#     img = sc.imread(Tr_list[idx])
#     img = np.double(sc.imresize(img, [height, width, channels], interp='bilinear', mode = 'RGB'))
#     Data_train_2018[idx, :,:,:] = img

#     b = Tr_list[idx]
#     # numbers = re.findall(r'\d+', b)
#     match = re.search(r'\d+', b)
#     number = str(match.group())
#     print('###########',b)
#     print('@@@@@@@@@@@@@@@@@',number)
#     # print('##############',numbers)
#     # b = b[len(b)-8: len(b)-4]




#     # print(b)
#     add = ("/data/fengxiao/UltraLight-VM-UNet/data/AD/GT/" + number +'.png')
#     # add = ("/data/fengxiao/UltraLight-VM-UNet/data/AD/GT/" + numbers[1] +'.png')  # Masks storage folder. The Mask type should be a black and white image of an 8-bit png (0 pixels for the background and 255 pixels for the target).
#     img2 = sc.imread(add)
#     img2 = np.double(sc.imresize(img2, [height, width], interp='bilinear'))
#     Label_train_2018[idx, :,:] = img2    
         
# print('Reading your dataset finished')

# ################################################################ Make the training, validation and test sets ########################################    
# Train_img      = Data_train_2018[0:train_number,:,:,:]  
# Validation_img = Data_train_2018[train_number:train_number+val_number,:,:,:]
# Test_img       = Data_train_2018[train_number+val_number:all,:,:,:]

# Train_mask      = Label_train_2018[0:train_number,:,:]
# Validation_mask = Label_train_2018[train_number:train_number+val_number,:,:]
# Test_mask       = Label_train_2018[train_number+val_number:all,:,:]


# np.save('data_train', Train_img)
# np.save('data_test' , Test_img)
# np.save('data_val'  , Validation_img)

# np.save('mask_train', Train_mask)
# np.save('mask_test' , Test_mask)
# np.save('mask_val'  , Validation_mask)






# -*- coding: utf-8 -*-
# Required: pip install pillow numpy h5py scipy

import os
import numpy as np
import glob
import re
from PIL import Image

# Parameters
height = 512     # 图像高度
width = 512      # 图像宽度
channels = 3     # RGB通道数

train_number = 300  # 训练集数量
val_number = 60     # 验证集数量
test_number = 62    # 测试集数量
all = int(train_number + val_number + test_number)

# 图像文件夹路径
image_folder = "/data/fengxiao/UltraLight-VM-UNet/data/DeepCrack/train_img"
gt_folder = "/data/fengxiao/UltraLight-VM-UNet/data/DeepCrack/train_lab"


# 获取图像路径列表
Tr_list = glob.glob(os.path.join(image_folder, '*.jpg'))
Tr_list.sort()  # 保证顺序一致

# 创建空数组用于存储图像和标签
Data_train_2018 = np.zeros([all, height, width, channels], dtype=np.float64)
Label_train_2018 = np.zeros([all, height, width], dtype=np.float64)

print('Reading')
print(f"总图片数: {len(Tr_list)}")

for idx in range(len(Tr_list)):
    print(f"正在读取第 {idx+1} 张图像")

    # 读取 image 图像
    img_path = Tr_list[idx]
    img = Image.open(img_path).convert("RGB")
    img = img.resize((width, height), Image.BILINEAR)
    img = np.array(img, dtype=np.float64)
    Data_train_2018[idx, :, :, :] = img

    # 提取图片编号
    # match = re.search(r'(\d+)', os.path.basename(img_path))
    match = os.path.splitext(os.path.basename(img_path))[0]
    # if match:
    #     number = match.group()
    # else:
    #     raise ValueError(f"无法从路径中提取编号: {img_path}")

    print('图像路径:', img_path)
    # print('编号提取:', number)

    # 读取对应的 ground truth 图像
    # gt_path = os.path.join(gt_folder, f"{number}.png")
    gt_path = os.path.join(gt_folder, f"{match}.png")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"找不到 ground truth 图像: {gt_path}")

    img2 = Image.open(gt_path).convert("L")  # 灰度图
    img2 = img2.resize((width, height), Image.BILINEAR)
    img2 = np.array(img2, dtype=np.float64)
    Label_train_2018[idx, :, :] = img2

print('数据集读取完成')

# 划分训练集、验证集、测试集
Train_img       = Data_train_2018[0:train_number, :, :, :]
Validation_img  = Data_train_2018[train_number:train_number+val_number, :, :, :]
Test_img        = Data_train_2018[train_number+val_number:all, :, :, :]

Train_mask      = Label_train_2018[0:train_number, :, :]
Validation_mask = Label_train_2018[train_number:train_number+val_number, :, :]
Test_mask       = Label_train_2018[train_number+val_number:all, :, :]

# 保存为 .npy 文件
np.save('data_train.npy', Train_img)
np.save('data_val.npy', Validation_img)
np.save('data_test.npy', Test_img)

np.save('mask_train.npy', Train_mask)
np.save('mask_val.npy', Validation_mask)
np.save('mask_test.npy', Test_mask)

print("✅ 数据保存完成！生成了以下文件：")
print("  data_train.npy, data_val.npy, data_test.npy")
print("  mask_train.npy, mask_val.npy, mask_test.npy")