import torch.nn as nn
import torch

# 官方预训练权重
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class Vgg16_net(nn.Module):
    def __init__(self,
                 num_classes: int = 2):
        super(Vgg16_net, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # (32-3+2)/1+1=32   32*32*64
            nn.BatchNorm2d(64),
            # inplace是否将计算得到的值覆盖之前的值，比如
            nn.ReLU(inplace=True),
            # 意思就是对从上层网络Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存，不用多存储其他变量

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # (32-3+2)/1+1=32    32*32*64
            # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(kernel_size=2, stride=2)   #(32-2)/2+1=16         16*16*64
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1, stride=1),  # Shortcut层，将输入的通道数调整为64
            # nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True)
        )

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)  # (32-2)/2+1=16         16*16*64

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16  16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            # (16-3+2)/1+1=16   16*16*128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (16-2)/2+1=8     8*8*128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # (8-3+2)/1+1=8   8*8*256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (8-2)/2+1=4      4*4*256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (4-3+2)/1+1=4    4*4*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (4-2)/2+1=2     2*2*512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2    2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2     2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            # (2-3+2)/1+1=2      2*2*512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # (2-2)/2+1=1      1*1*512
        )

        self.conv = nn.Sequential(
            # self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

        self.fc = nn.Sequential(
            # y=xA^T+b  x是输入,A是权值,b是偏执,y是输出
            # nn.Liner(in_features,out_features,bias)
            # in_features:输入x的列数  输入数据:[batchsize,in_features]
            # out_freatures:线性变换后输出的y的列数,输出数据的大小是:[batchsize,out_features]
            # 定义分类网络模块结构，就是三层全连接层（nn.Sequential函数用来定义模型、包装网络层）
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

        init_weights = True
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():      # 遍历网络的每一层
            if isinstance(m, nn.Conv2d):    # 遍历到卷积层时
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)   # 使用Xavier方法初始化卷积核的权重参数
                if m.bias is not None:    # 如果该卷积核采用了bias偏置，就将该bias初始化为0
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):   # 遍历到全连接层时
                nn.init.xavier_uniform_(m.weight)
                # nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    def _initialize_weights1(self):
        for m in self.modules():      # 遍历网络的每一层
            if isinstance(m, nn.Sequential) and hasattr(m, "shortcut"):    # 遍历到卷积层时
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight)   # 使用Xavier方法初始化卷积核的权重参数
                if m.bias is not None:    # 如果该卷积核采用了bias偏置，就将该bias初始化为0
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.Linear):   # 遍历到全连接层时
            #     nn.init.xavier_uniform_(m.weight)
            #     # nn.init.normal_(m.weight, 0, 0.01)
            #     nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out1 = self.layer1(x)
        out_shortcut = self.shortcut(x)
        x = out1 + out_shortcut
        x = self.maxpool(x)
        # print(x.shape)
        x = nn.ReLU(inplace=True)(x)  # ReLU激活函数
        x = self.conv(x)

        x = torch.flatten(x, start_dim=1)  # 展平操作，从第一个维度进行展平处理（因为第0个维度是batch）
        x = self.fc(x)
        return x


# model = Vgg16_net(num_classes=2)  # 将模型确定到指定的设备上
# # summary(model, (3, 224, 224))
# state_dict = model.state_dict()  # 获取模型的状态字典
# # 输出layer1的权重参数名称
# for name in state_dict:
#     if name.startswith('layer1'):
#         print(name)
#     if name.startswith('shortcut'):
#         print("-------------")
#         print(name)
#     if name.startswith('maxpool'):
#         print("---------------")
#         print(name)

