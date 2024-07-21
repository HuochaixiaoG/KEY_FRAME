import sys
import json
import pickle
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix, precision_score, f1_score


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 读取分类数据（目录级别较少时用）
def read_split_data_2(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 打开root文件
    dirs = os.listdir(root)
    print("dirs:{}".format(dirs))
    # 遍历文件夹，一个文件夹对应一个类别
    image_class = [clas for clas in os.listdir(root) if os.path.isdir(os.path.join(root, clas))]
    # 排序，保证各平台顺序一致
    image_class.sort()
    print("image_class：{}".format(image_class))
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(image_class))          # 将image_class映射为字典
    print("class_indices：{}".format(class_indices))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    print("json_str：{}".format(json_str))
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    # # 以上两句可以用以下一句代替
    # json.dump(dict((val, key) for key, val in class_indices.items()), open('class_indices.json', "w"))

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in image_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        img_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))
        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(img_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(img_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    plot_image = True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(0, 2), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(image_class)), image_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('')
        # 设置y坐标
        plt.ylabel('number')
        # 设置柱状图的标题
        plt.title('Sample category Chart')
        # plt.show()

    # 调用绘制数据分布图的函数
    plot_data_distribution(train_images_label, val_images_label, class_indices)

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 打开root文件
    dirs = os.listdir(root)
    # 输出所有文件和文件夹
    for file in dirs:
        print("\n-----------------root所有文件夹 ：------------------")
        print(file)

    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [clas for clas in os.listdir(root) if os.path.isdir(os.path.join(root, clas))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    print("\n-------------------flower_class排序------------------")
    print(flower_class)
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))          # 将flower_class映射为字典
    print("\n--------class_indices类别索引----------------")
    print(class_indices)

    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    print("\n--------json_str----------------")
    print(json_str)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for clas in flower_class:
        print("\n=================clas、clas_path================")
        clas_path = root + "\\" + clas
        print(clas)       # clas:celeb-real、celeb-synthesis
        print(clas_path)   # D:\Deepfakes_datasets\celeb-df-v1-img\celeb-real、  D:\Deepfakes_datasets\celeb-df-v1-img\celeb-synthesis
        images_number = 0
        for cla in os.listdir(clas_path):     # cla是celeb-real/celeb-synthesis下的文件夹
            cla_path = os.path.join(clas_path, cla)
            # 遍历获取supported支持的所有文件路径
            images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]           # images是imgs文件夹内的所有图片
            # 排序，保证各平台顺序一致
            images.sort()
            # 获取该类别对应的索引
            image_class = class_indices[clas]
            # 记录该类别的样本数量
            images_number = len(images) + images_number    # 将每个子文件夹内的图片个数相加
            # 按比例随机采样验证样本
            val_path = random.sample(images, k=int(len(images) * val_rate))

            for img_path in images:
                if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                    val_images_path.append(img_path)
                    val_images_label.append(image_class)
                else:  # 否则存入训练集
                    train_images_path.append(img_path)
                    train_images_label.append(image_class)
        every_class_num.append(images_number)
        print("每个种类的样本个数:" + str(every_class_num))

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."

    # 画数据类别数量图
    plot_image = True
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(0, 2), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(flower_class)), flower_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('')
        # 设置y坐标
        plt.ylabel('number')
        # 设置柱状图的标题
        plt.title('Sample category Chart')
        plt.show()

    # 调用绘制数据分布图的函数
    plot_data_distribution(train_images_label, val_images_label, class_indices)

    return train_images_path, train_images_label, val_images_path, val_images_label

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()

# 绘制每个类别的训练集和验证集数据量柱状图
def plot_data_distribution(train_labels, val_labels, class_indices):
    train_class_counts = [0] * len(class_indices)
    val_class_counts = [0] * len(class_indices)

    for label in train_labels:
        train_class_counts[label] += 1

    for label in val_labels:
        val_class_counts[label] += 1

    x = range(len(class_indices))
    plt.bar(x, train_class_counts, width=0.4, align='center', label='Train')
    plt.bar(x, val_class_counts, width=0.4, align='edge', label='Validation')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks(x, class_indices.values())
    plt.title('Data Distribution')
    plt.legend()
    plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()   # 交叉熵损失
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()    # 梯度归零
    sample_num = 0

    data_loader = tqdm(data_loader, file=sys.stdout)      # tqdm模块是python进度条库

    pred_probs = []
    true_labels = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]    # img.shape[0]：图像的垂直尺寸（高度）

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]      # torch.max(pred, dim=1)表示pred中每行最大值的值和索引
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()   # torch.eq()比较对应位置数字，相同则为1，否则为0。.sum()就是将所有值相加
        loss = loss_function(pred, labels.to(device))
        loss.backward()     # 反向传播计算得到每个参数的梯度值
        accu_loss += loss.detach()

        # 计算AUC、精确度（precision）、F1值以及混淆矩阵
        pred_classes = torch.argmax(torch.softmax(pred, dim=1), dim=1).detach().cpu().numpy()
        pred_probs.extend(pred_classes)
        true_labels.extend([label.item() for label in labels])
        if len(np.unique(true_labels)) > 1:
            auc = roc_auc_score(true_labels, pred_probs)
            precision = precision_score(true_labels, pred_probs, zero_division=1)
            f1 = f1_score(true_labels, pred_probs)
            confusion = confusion_matrix(true_labels, np.array(pred_probs).round())
        else:
            auc = precision = f1 = 0.0
            confusion = np.zeros((2, 2))

        # desc: 字符串, 进度条左边的描述文字
        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}, AUC: {:.4f}, Precision: {:.4f}, F1: {:.4f}, TP: {}, TN: {}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, f1,
            confusion[1, 1], confusion[0, 0])

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()     # 通过随机梯度下降执行一步参数更新
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num

@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()      # 交叉熵损失函数
    model.eval()
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)

    pred_probs = []
    true_labels = []

    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()
        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        # 计算AUC、精确度（precision）、F1值以及混淆矩阵
        pred_classes = torch.argmax(torch.softmax(pred, dim=1), dim=1).detach().cpu().numpy()
        pred_probs.extend(pred_classes)
        true_labels.extend([label.item() for label in labels])
        if len(np.unique(true_labels)) > 1:
            auc = roc_auc_score(true_labels, pred_probs)
            precision = precision_score(true_labels, pred_probs, zero_division=1)
            f1 = f1_score(true_labels, pred_probs)
            confusion = confusion_matrix(true_labels, np.array(pred_probs).round())
        else:
            auc = precision = f1 = 0.0
            confusion = np.zeros((2, 2))

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}, AUC: {:.4f}, Precision: {:.4f}, F1: {:.4f}, TP: {}, TN: {}".format(
            epoch, accu_loss.item() / (step + 1), accu_num.item() / sample_num, auc, precision, f1, confusion[1, 1], confusion[0, 0])

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def read_split_data_3(root: str, val_rate: float = 0.3):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    dirs = os.listdir(root)
    flower_class = [clas for clas in dirs if os.path.isdir(os.path.join(root, clas))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型

    for clas in flower_class:
        clas_path = os.path.join(root, clas)
        images_number = 0
        for cla in os.listdir(clas_path):
            cla_path = os.path.join(clas_path, cla)
            images = [os.path.join(cla_path, i) for i in os.listdir(cla_path)
                      if os.path.splitext(i)[-1] in supported]
            images.sort()
            image_class = class_indices[clas]
            images_number += len(images)

            train_paths, val_paths, train_labels, val_labels = train_test_split(images, [image_class] * len(images),
                                                                                test_size=val_rate, random_state=0,
                                                                                stratify=[image_class] * len(images))
            train_images_path.extend(train_paths)
            train_images_label.extend(train_labels)
            val_images_path.extend(val_paths)
            val_images_label.extend(val_labels)

        every_class_num.append(images_number)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must be greater than 0."
    assert len(val_images_path) > 0, "number of validation images must be greater than 0."

    # 画数据类别数量图
    plot_image = True
    if plot_image:
        plt.bar(range(0, 2), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('')
        plt.ylabel('number')
        plt.title('Sample category Chart')
        plt.show()

    # 调用绘制数据分布图的函数
    plot_data_distribution(train_images_label, val_images_label, class_indices)

    return train_images_path, train_images_label, val_images_path, val_images_label
