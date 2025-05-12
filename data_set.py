import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# import data_preprocess
# 定义标签列表
LABEL_LIST = ["normal", "ill"]  # 定义标签列表

# 自定义数据集类
class MyData(Dataset):
    def __init__(self, root_dir, label_dir, transform=None):  # 定义类的初始化方法，接收三个参数：root_dir, label_dir, transform
        self.root_dir = root_dir  # 将传入的root_dir参数赋值给实例变量self.root_dir，用于存储根目录路径
        self.label_dir = label_dir  # 将传入的label_dir参数赋值给实例变量self.label_dir，用于存储标签目录路径
        self.path = os.path.join(self.root_dir, self.label_dir)  # 使用os.path.join将根目录和标签目录拼接成完整路径，并赋值给实例变量self.path
        self.img_path = os.listdir(self.path)  # 使用os.listdir获取self.path目录下的所有文件名，并赋值给实例变量self.img_path
        self.transform = transform  # 将传入的transform参数赋值给实例变量self.transform，用于存储图像转换方法（如果有的话）

    def __getitem__(self, idx):
        # 获取指定索引idx对应的图片名称
        img_name = self.img_path[idx]
        # 构建图片的完整路径，由根目录、标签目录和图片名称组成
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        
        #因为使用的res模型需要三维输入，但现有的数据都是二维黑白两色，所以需要将其转换成三维RGB
        # 如果有定义transform，则对图片进行预处理
        if self.transform:
            # 打开图片并将其转换为RGB格式
            img = Image.open(img_item_path).convert('RGB')
            # 应用transform进行图片预处理
            img = self.transform(img)
        else:
            # 打开图片并将其转换为RGB格式
            img = Image.open(img_item_path).convert('RGB')
            # 如果没有定义transform，则将图片转换为Tensor格式
            img = transforms.ToTensor()(img)
            
        # img=data_preprocess.image_augment(img)
        
        # 获取图片对应的标签，标签由标签目录在标签列表中的索引表示
        label = LABEL_LIST.index(self.label_dir)
        # 将标签转换为torch tensor格式
        label = torch.tensor(label)
        # 返回处理后的图片和对应的标签
        return img, label

    def __len__(self):
        return len(self.img_path)

# 获取数据集
def get_datasets(myroot_dir="data", train_transform=None, val_transform=None):
    ill_dir = "ill"
    normal_dir = "normal"

    ill_dataset = MyData(myroot_dir, ill_dir, transform=train_transform)
    normal_dataset = MyData(myroot_dir, normal_dir, transform=train_transform)

    myroot_dir += "/test"
    ill_dataset_test = MyData(myroot_dir, ill_dir, transform=val_transform)
    normal_dataset_test = MyData(myroot_dir, normal_dir, transform=val_transform)

    train_set = ill_dataset + normal_dataset
    val_set = ill_dataset_test + normal_dataset_test

    return train_set, val_set