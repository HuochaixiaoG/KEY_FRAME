import torch.nn as nn
import torch

# 官方预训练权重
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=2, init_weights=True):   # init_weights=False不初始化权重参数
        super(VGG, self).__init__()
        self.features = features
        # 定义分类网络模块结构，就是三层全连接层（nn.Sequential函数用来定义模型、包装网络层）
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.features(x)
        # N x 512 x 7 x 7
        x = torch.flatten(x, start_dim=1)   # 展平操作，从第一个维度进行展平处理（因为第0个维度是batch）
        # N x 512*7*7
        x = self.classifier(x)
        return x

    # 初始化权重参数
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

# 定义函数——生成特征提取的网络结构
def make_features(cfg: list):
    layers = []    # 用于存放创建的每一层的结构
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)   # stride默认是1，所以这里没有特别设置stride参数
            layers += [conv2d, nn.ReLU(True)]
            in_channels = v
    # 通过for循环的layers叠加，return网络结构的所有层
    return nn.Sequential(*layers)

# M指maxpool，64、128指卷积核个数
cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

# 实例化模型
def vgg(model_name="vgg16", **kwargs):
    assert model_name in cfgs, "Warning: model number {} not in cfgs dict!".format(model_name)
    cfg = cfgs[model_name]

    model = VGG(make_features(cfg), **kwargs)     # 将make_features函数的输出作为VGG类的feature参数传入
    state_dict = model.state_dict()  # 获取模型的状态字典
    # 输出layer1的权重参数名称
    for name in state_dict:
        # print(name)
        if name.startswith('features'):
            print(name)
    print("-------已实例化vgg模型-----------")
    return model
