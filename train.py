import os
import math
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import transforms
import torch.optim.lr_scheduler as lr_scheduler
from my_dataset import MyDataSet
from utils import read_split_data, train_one_epoch, evaluate
from torchsummary import summary
# from model import vgg as create_model
# from Vgg16_3 import VGG16 as create_model

# from vgg16_2 import Vgg16_net as create_model
# from EffiV2S import efficientnetv2_s as create_model
# from ResNet import resnet50 as create_model
# from mobilenet import MobileNetV2 as create_model
# from Xception import xception as create_model
# from EfficientNet import efficientnet_b0 as create_model
from Xception_convlstm import modified_xception as create_model

# print(torch.__version__)
# print(torch.version.cuda)

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")  # 指定使用的设备，若有GPU设备则使用GPU，若没有，则使用CPU
    print("using {} device.".format(device))
    print(args)
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    tb_writer = SummaryWriter()     # 保存在默认文件夹：runs/...
    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)


    img_size = {"s": [224, 224],  # train_size, val_size最大值是[300, 384]，可设置为更低的值
                "m": [384, 480],
                "l": [384, 480]}
    num_model = "s"

    data_transform = {
        # 对于训练集
        "train": transforms.Compose([transforms.Resize(img_size[num_model][0]),  # 进行随机裁剪成num_model大小
                                     transforms.RandomRotation(degrees=15, fill=(0, 0, 255)), # 随机旋转，以一定概率对图像进行随机旋转，旋转角度在-15到+15之间
                                     transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                    #  AddGaussianNoise(mean=random.uniform(0.5, 1.5), variance=0.5, amplitude=random.uniform(0, 45)),  # 自定义的高斯噪声
                                     transforms.RandomHorizontalFlip(p=0.2),  # 以30%的概率对图像进行水平翻转
                                     transforms.RandomPerspective(distortion_scale=0.3, p=0.1, fill=0),  # 随机扭曲
                                     transforms.ToTensor(),  # 转化成tensor
                                     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # 进行标准化处理
                                     ]),

        # 对于验证集
        "val": transforms.Compose([transforms.Resize(img_size[num_model][0]),
                                   # transforms.CenterCrop(img_size[num_model][0]),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                                   ])
    }
    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))    # 每个进程使用{}个数据加载器工作线程
    # 获取一批批的数据
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,  # 使用 shuffle=True 参数来打乱数据顺序
                                               pin_memory=True,
                                               num_workers=nw,  # num_workers表示加载数据需要的线程个数，0表示使用主线程加载数据
                                               collate_fn=train_dataset.collate_fn
                                               )
    # 载入测试集
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn
                                             )
    # # 使用vgg模型
    # model_name = "vgg16"
    # model = create_model(model_name=model_name, num_classes=2).to(device)
    # # summary(model, (3, 224, 224))

    model = create_model(num_classes=2).to(device)
    # summary(model, (3, 224, 224))

    # 如果存在预训练权重则载入
    if args.weights != "":
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location=device)
            # 加载原模型的与训练权重
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            print(model.load_state_dict(load_weights_dict, strict=False))
        else:
            print("不存在预训练权重")
            raise FileNotFoundError("not found weights file: {}".format(args.weights))

    # freeze_layers=True,则使用预训练权重并且冻结某些层的权重
    if args.freeze_layers:
        print("冻结权重------------")
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结(head:Sequential)
            if "head" not in name:
                para.requires_grad_(False)   # False冻结此层参数，不影响误差反向传播正常进行，但是权重和偏置值不更新
                print("head not in name:{}".format(name))
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=0.01)     # 权重衰减率0.0001
    optimizer = optim.Adam(pg, lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)  # Adam优化算法，betas是一阶、二阶矩估计的指数衰减率

    # 定义一个lambda函数lf，根据cosine策略计算每个epoch的学习率，输入参数是x,表示当前epoch的编号
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # math.cos返回余弦值
    # 使用LambdaLR方法定义一个学习率调节器，是optimizer和lr_lambda，分别表示优化器和计算学习率的函数
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # 定义了一些变量
    min_loss = 100000.0
    train_min_loss = 100000.0
    max_acc = 0.1
    train_max_acc = 0.1
    best_loss_epoch = 0
    best_acc_epoch = 0
    train_best_loss_epoch = 0
    train_best_acc_epoch = 0
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(args.epochs):
        # print("..............................................................")
        # 进行训练
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()     # 调用计算学习率的函数 (lr_lambda) 并根据函数的返回值更新优化器的学习率。

        # validate 测试
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)    # SummaryWriter.add_scalar书写坐标图（标题，y,x)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))    # 保存模型的权重参数
        if val_loss <= min_loss:
            min_loss = val_loss
            best_loss_epoch = epoch
        if val_acc >= max_acc:
            max_acc = val_acc
            best_acc_epoch = epoch
        if train_loss <= train_min_loss:
            train_min_loss = train_loss
            train_best_loss_epoch = epoch
        if train_acc >= train_max_acc:
            train_max_acc = train_acc
            train_best_acc_epoch = epoch
        train_loss_list.append(format(train_loss, '.3f'))
        val_loss_list.append(format(val_loss, '.3f'))
        train_acc_list.append(format(train_acc, '.3f'))
        val_acc_list.append(format(val_acc, '.3f'))
    tb_writer.close()
    print("训练集loss_min:{:.3f}, epoch:{}".format(train_min_loss, train_best_loss_epoch))
    print("训练集acc_max:{:.3f}, epoch:{}".format(train_max_acc, train_best_acc_epoch))
    print("验证集loss_min:{:.3f}, epoch:{}".format(min_loss, best_loss_epoch))
    print("验证集acc_max:{:.3f}, epoch:{}".format(max_acc, best_acc_epoch))
    print("train_acc = {}".format(train_acc_list))
    print("train_loss = {}".format(train_loss_list))
    print("valid_acc = {}".format(val_acc_list))
    print("valid_loss = {}".format(val_loss_list))

    plt.plot(np.arange(len(train_loss_list)), list(map(float, train_loss_list)), color="orange", label="train loss")
    plt.plot(np.arange(len(train_acc_list)), list(map(float, train_acc_list)), color="red", label="train acc")
    plt.plot(np.arange(len(val_loss_list)), list(map(float, val_loss_list)), color="green", label="valid loss")
    plt.plot(np.arange(len(val_acc_list)), list(map(float, val_acc_list)), color="blue", label="valid acc")
    plt.legend()  # 显示图例
    plt.xlabel('epoches')
    plt.ylabel("{}".format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    plt.title('Accuracy & loss')
    plt.savefig("./loss_and_acc_{}.jpg".format(time.strftime('%Y-%m-%d %H_%M_%S', time.localtime())))
    plt.show()

#添加高斯噪声
from PIL import Image
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0, p=0.5):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img

if __name__ == '__main__':
    parser = argparse.ArgumentParser()         # 使用argparse 模块的第一步是创建一个 ArgumentParser 对象
    parser.add_argument('--num_classes', type=int, default=2)     # 通过调用add_argument()方法给ArgumentParser添加程序参数信息
    parser.add_argument('--epochs', type=int, default=100)         # default是不指定参数时的默认值
    parser.add_argument('--batch-size', type=int, default=32)      # type是命令行参数应该被转换成的类型
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lrf', type=float, default=0.01)
    # 数据集所在根目录
    parser.add_argument('--data-path', type=str, default=r"Z:\XDG\DFDC_opticalflow_weighted_keyframes_0.3")
                                                         #Z:\XDG\celeb-df-v1_opticalflow_weighted_keyframes_0.5
                                                         #Z:\XDG\FF++_opticalflow_weighted_keyframes_0.5
                                                         #Z:\XDG\DFDC\DFDC_opticalflow_weighted_keyframes_0.5
                                                         #Z:\XDG_XCEPTION\FF++_opticalflow_weighted_keyframes_0.3
                                                         #Z:\XDG\Celeb-df_opticalflow_weighted_keyframes_0.3
    # 模型权重
    parser.add_argument('--weights', type=str,default=r'C:\Users\Karbei\Desktop\0619.pth',
                        # default=r'C:\Users\Karbei\Desktop\0310_1.pth',
                        #C:\Users\Karbei\Desktop\0301_1.pth
                        #C:\Users\Karbei\Desktop\1115EffiV2S_1.pth
                        # default='D:\\Deepfake_detection_codes\\VGG16-pytorch\\weights\\efficientnetv2-s.pth',
                        # default='D:\\Deepfake_detection_codes\\VGG16-pytorch\\weights\\vgg16.pth',
                        help='initial weights path')
    parser.add_argument('--freeze-layers', type=bool, default=False)    # 如果不使用'./pre_efficientnetv2-s.pth预训练的权重，无需冻结某层，就设置为False
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()    # 使用 parse_args() 解析添加的参数
    main(opt)
