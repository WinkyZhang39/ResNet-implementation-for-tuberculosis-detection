import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision.transforms import ToPILImage, ToTensor

def random_brightness(img, low=0.5, high=1.5):
    ''' 随机改变亮度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Brightness(img).enhance(x)
    return img

def random_contrast(img, low=0.5, high=1.5):
    ''' 随机改变对比度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Contrast(img).enhance(x)
    return img

def random_color(img, low=0.5, high=1.5):
    ''' 随机改变饱和度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Color(img).enhance(x)
    return img

def random_sharpness(img, low=0.5, high=1.5):
    ''' 随机改变清晰度(0.5~1.5) '''
    x = random.uniform(low, high)
    img = ImageEnhance.Sharpness(img).enhance(x)
    return img

def random_noise(img, low=0, high=10):
    ''' 随机加高斯噪声(0~10) '''
    img = np.asarray(img)
    if img.ndim == 2:  # 如果是灰度图像，转换为RGB图像
        img = np.stack([img] * 3, axis=-1)
    sigma = np.random.uniform(low, high)
    noise = np.random.randn(*img.shape[:2], 3) * sigma  # 生成与图像形状相同的噪声
    img = img + np.round(noise).astype('uint8')
    img = np.clip(img, 0, 255)  # 将矩阵中的所有元素值限制在0~255之间
    img = Image.fromarray(img)
    return img

def image_augment(img, prob=0.5):
    ''' 叠加多种数据增强方法 '''
    if isinstance(img, Image.Image):  # 如果是PIL.Image对象，转换为ndarray
        img = np.array(img)
    opts = [random_brightness, random_contrast, random_color, random_sharpness]
    random.shuffle(opts)
    for opt in opts:
        if random.random() < prob:
            img = opt(img)  # 处理图像
    return img
