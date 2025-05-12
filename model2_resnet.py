import torch.nn as nn
from torchvision import models

# class ClassificationModelResNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ClassificationModelResNet, self).__init__()
#         # 加载预训练ResNet-50并冻结部分层（可选）
#         self.resnet = models.resnet50(pretrained=True)  # 更深的网络结构
        
#         # 冻结除最后一层外的所有参数（根据需求选择）
#         for param in self.resnet.parameters():
#             param.requires_grad = False  # 冻结所有层
        
#         # 替换全连接层（添加中间层和正则化）
#         self.resnet.fc = nn.Sequential(
#             nn.Dropout(0.5),                  # 防止过拟合
#             nn.Linear(2048, 256),             # 修改输入特征维度为2048
#             nn.ReLU(),                        # 非线性激活
#             nn.Linear(256, num_classes)        # 最终分类层
#         )
# # 修改模型初始化代码（model2_resnet.py）
# class ClassificationModelResNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ClassificationModelResNet, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
        
#         # 仅冻结前3个残差块（layer1-3）
#         for name, param in self.resnet.named_parameters():
#             if 'layer4' not in name and 'fc' not in name:
#                 param.requires_grad = False
        
#         self.resnet.fc = nn.Linear(512, num_classes)
             
#     def forward(self, x):
#         return self.resnet(x)

# class ClassificationModelResNet(nn.Module):
#     def __init__(self, num_classes=2):
#         super(ClassificationModelResNet, self).__init__()
#         self.resnet = models.resnet18(pretrained=True)
        
#         # 仅冻结前3个残差块（layer1-3）
#         for name, param in self.resnet.named_parameters():
#             if 'layer4' not in name and 'fc' not in name:
#                 param.requires_grad = False
        
#         # 添加Dropout层
#         self.dropout = nn.Dropout(0.5)  # 可以调整丢弃率
        
#         # 修改全连接层
#         self.resnet.fc = nn.Sequential(
#             nn.Linear(512, 256),
#             self.dropout,
#             nn.ReLU(),
#             # nn.Dropout(0.3),
#             nn.Linear(256, num_classes)
#         )
             
#     def forward(self, x):
#         return self.resnet(x)


class ClassificationModelResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 使用预训练ResNet
        self.resnet = models.resnet50(pretrained=True) 
        
        # 冻结前三个stage的参数
        for param in self.resnet.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
            
        num_ftrs = self.resnet.fc.in_features
        # 添加Dropout层
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.resnet(x)