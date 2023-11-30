# conda create --name MOSCOWCODE

# conda activate MOSCOWCODE
# python script_version_neuro_wm.py

# https://www.hse.ru/edu/vkr/833281172 # Встраивание цифровых водяных знаков в изображения на основе нейронных сетей 
# https://www.microsoft.com/en-us/download/details.aspx?id=54765 # Kaggle Cats and Dogs Dataset

# pip install --upgrade pip
# pip install notebook
# pip install opencv-python
# pip install matplotlib

# pip install torch
# pip install torchvision
## vs.
# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
## vs.
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


# pip install typing
# pip install ipython
# pip install ipywidgets
# pip install tqdm

# jupyter notebook 

# conda deactivate


'''
>> python script_version_neuro_wm.py
>> device = cpu
>> Введите путь и имя изображения-контейнера с расширением: test3.png
>> Введите путь и имя изображения-watermark с расширением: wm.jpg
>> Введите путь и имя атакуемого изображения с расширением: test3.png
'''

# Подключение библиотек 
import numpy as np 
import cv2 as cv 
import matplotlib.pyplot as plt 
from PIL import Image 
import torch 
from torch import nn, optim 
import torchvision.transforms as transforms 
from typing import Type, Union 
from IPython.display import clear_output, display 
from ipywidgets import Output 
from tqdm.auto import trange 
from numpy.random import randint 
import os 
import zipfile 
from torch.utils.data import Dataset 
from torchvision import datasets 
 

device = "cuda" if torch.cuda.is_available() else "cpu" 
print(f"device = {device}")


# Класс преобразования контейнера 
class PreprocessImage: 
    # Получение информации о границах изображения 
    def edge_information(self, image): 
        img_np = np.array(image*255).transpose(1, 2, 0).astype(np.uint8) 
        canny = cv.Canny(img_np,100,200) 
        tau = 2 
        edge = (canny + 1) / tau 
        edge = np.exp(edge * (-1)) 
        return torch.from_numpy(edge) 

    # Получение информации о цветности изображения 
    def chrominance_information(self, image): 
        new_img = image #* 255 
        y = 0.299 * new_img[0] + 0.587 * new_img[1] + 0.114 * new_img[2] 
        cb = 0.564*(new_img[2] - y)  
        cr = 0.713*(new_img[0] - y) 
        teta = 0.25 
        cb_norm = torch.square(cb) 
        cr_norm = torch.square(cr) 
        chrominance = (cb_norm + cr_norm) / (teta ** 2) * (-1) 
        chrominance = torch.exp(chrominance) * (-1) + 1 
        return chrominance
    
    # Преобразование изображения 
    def preprocess_cover(self, image): 
        img_norm = torch.zeros(image.size()) 
        chrominance = self.chrominance_information(image) 
        edge = self.edge_information(image) 
        know = (chrominance + edge) / 2 
        img_norm[0] = image[0] + know - 1 
        img_norm[1] = image[1] + know - 1 
        img_norm[2] = image[2] + know - 1 
        return img_norm 


# Класс кодировщика 
class Encoder(nn.Module): 
    # Инициализация слоев нейросети 
    def __init__(self): 
        super(Encoder, self).__init__() 
         
        self.conv1_watermark = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1) 
        self.conv2_watermark = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv3_watermark = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv4_watermark = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv5_watermark = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv6_watermark = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv7_watermark = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv1_cover = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        self.conv2_cover = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) 
        self.conv3_cover = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv4_cover = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) 
        self.conv5_cover = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv6_cover = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) 
        self.conv7_cover = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1) 
        self.conv8_cover = nn.Conv2d(in_channels=35, out_channels=64, kernel_size=3, padding=1) 
        self.conv9_cover = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.conv9_1_cover = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) 
        self.conv9_2_cover = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1) 
        self.conv10_cover = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) 
        self.conv11_cover = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1) 
        self.conv12_cover = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) 
        self.conv13_cover = nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1) 
 
        self.activator = nn.ReLU() 

# Структура нейросети 
    def forward(self, input): 
        (cover, watermark, cover_orig) = input 
 
        watermark = self.conv1_watermark(watermark) 
        cover = self.conv1_cover(cover) 
 
        cover = torch.cat([cover, watermark], 1) 
 
        watermark = self.conv2_watermark(watermark) 
        watermark = self.conv3_watermark(watermark) 
        cover = self.conv2_cover(cover) 
        cover = self.conv3_cover(cover) 
 
        cover = torch.cat([cover, watermark], 1) 
 
        watermark = self.conv4_watermark(watermark) 
        watermark = self.conv5_watermark(watermark) 
        cover = self.conv4_cover(cover) 
        cover = self.conv5_cover(cover) 
 
        cover = torch.cat([cover, watermark], 1) 
 
        watermark = self.conv6_watermark(watermark) 
        watermark = self.conv7_watermark(watermark) 
        cover = self.conv6_cover(cover) 
        cover = self.conv7_cover(cover) 
 
        cover = torch.cat([cover, watermark, cover_orig], 1) 
 
        cover = self.conv8_cover(cover) 
        cover = self.activator(self.conv9_cover(cover)) 
        cover = self.activator(self.conv9_1_cover(cover)) 
        cover = self.activator(self.conv9_2_cover(cover)) 
        cover = self.activator(self.conv10_cover(cover)) 
        cover = self.activator(self.conv11_cover(cover)) 
        cover = self.activator(self.conv12_cover(cover)) 
        cover = self.conv13_cover(cover) 
 
        return cover 

# Класс декодера 
class Decoder(nn.Module): 
    # Инициализация слоев нейросети 
    def __init__(self): 
        super(Decoder, self).__init__() 
         
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1) 
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1) 
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1) 
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1) 
         
        self.bn1 = nn.BatchNorm2d(16) 
        self.bn2 = nn.BatchNorm2d(32) 
        self.bn3 = nn.BatchNorm2d(64) 
        self.bn4 = nn.BatchNorm2d(128) 
        self.bn5 = nn.BatchNorm2d(64) 
        self.bn6 = nn.BatchNorm2d(32) 
        self.bn7 = nn.BatchNorm2d(16) 
 
        self.activator = nn.ReLU() 
 
    # Структура нейросети 
    def forward(self, input): 
        output = self.activator(self.bn1(self.conv1(input)))       
        output = self.activator(self.bn2(self.conv2(output))) 
        output = self.activator(self.bn3(self.conv3(output)))                           
        output = self.activator(self.bn4(self.conv4(output)))      
        output = self.activator(self.bn5(self.conv5(output))) 
        output = self.activator(self.bn6(self.conv6(output))) 
        output = self.activator(self.bn7(self.conv7(output))) 
        output = self.conv8(output)
        return output 

# Класс дискриминатора 
class Discriminator(nn.Module): 
    # Инициализация слоев нейросети 
    def __init__(self): 
        super(Discriminator, self).__init__() 
 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) 
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1) 
         
        self.bn1 = nn.BatchNorm2d(16) 
        self.bn2 = nn.BatchNorm2d(32) 
        self.bn3 = nn.BatchNorm2d(64) 
        self.bn4 = nn.BatchNorm2d(128) 
        self.bn5 = nn.BatchNorm2d(256) 
 
        self.activator = nn.ReLU() 
        self.sigmoid = nn.Sigmoid() 
        self.pool = nn.AvgPool2d(2) 
        self.fc = nn.Linear(256 * 64 * 64, 1) 
 
    # Структура нейросети 
    def forward(self, input): 
        output = self.activator(self.bn1(self.conv1(input)))       
        output = self.activator(self.bn2(self.conv2(output))) 
        output = self.activator(self.bn3(self.conv3(output))) 
        output = self.activator(self.bn4(self.conv4(output))) 
        output = self.activator(self.bn5(self.conv5(output))) 
        output = self.pool(output) 
        output = output.view(-1, 256 * 64 * 64) 
        output = self.fc(output) 
        output = self.sigmoid(output) 
 
        return output 

# Класс симулятора атаки 
class Attack: 
    # Размытие по Гауссу 
    def gaussian(self, image, p=3): 
        transform_gaussian = transforms.Compose([transforms.GaussianBlur(p)]) 
        return transform_gaussian(image) 
 
    # Кадрирование 
    def cropping(self, image): 
        crop = torch.ones(image.size()).to(device) 
        a = randint(0,crop.shape[1]-40) 
        c = randint(0,crop.shape[2]-40) 
        crop[:,a:a+40,c:c+40] = 0 
        return image * crop 
 
    # Выбивание пикселей 
    def dropout(self, image, p=0.15): 
        mask = np.random.choice([0,1],image.size()[1:],True,[p,1-p]) 
        mask = torch.from_numpy(mask).to(device) 
        return image[:] * mask 
 
        # Выбивание пикселей 
    def salt(self, image, p=0.2): 
        salt = np.random.choice([0,1],image.size()[1:],True,[p/2,1-p/2]) 
        peper = np.random.choice([0,1],image.size()[1:],True,[1-p/2,p/2]) 
        salt = torch.from_numpy(salt).to(device) 
        peper = torch.from_numpy(peper).to(device) 
        return image[:] * salt + peper 
 
    def medianFilter(self, image, p = 5): 
        img_np = np.asarray(image.cpu().detach()).transpose(1,2,0) 
        img_bl = cv.medianBlur(img_np, p) 
        return transforms.ToTensor()(img_bl) 
 
    def jpg(self, image, p=90): # 90% сжатие
        img_np = np.asarray(image.cpu().detach()*255).transpose(1,2,0) 
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), p] # For JPEG, it can be a quality from 0 to 100 (the higher is the better). Default value is 95.
        result, encimg = cv.imencode('.jpg', img_np, encode_param) 
        decimg = cv.imdecode(encimg, 1) 
        return transforms.ToTensor()(decimg) 
 
    # Случайная атака 
    def random_attack(self, image): 
        attack = randint(0,7) 
        if attack == 1: 
            return self.gaussian(image) 
        elif attack == 2: 
            return self.cropping(image) 
        elif attack == 3: 
            return self.dropout(image) 
        elif attack == 4: 
            return self.salt(image) 
        elif attack == 5:
            return self.medianFilter(image) 
        elif attack == 6: 
            return self.jpg(image)    
        return image

# Класс автокодировщика 
class AutoEncoder(nn.Module): 
    # Инициализация переменных 
    def __init__(self) -> None: 
        super().__init__() 
 
        self.encoder = Encoder() 
        self.decoder = Decoder() 
        self.discriminator = Discriminator() 
        self.attack_class = Attack() 
        self.alfa = 0.5 
        self.beta = 0.5 
        self.sigma = 0.001 
        self.criterion = nn.MSELoss() 
 
    # Кодирование 
    def encode(self, x, y, z): 
        return self.encoder((x,y,z)) 
 
    # Декодирование 
    def decode(self, x): 
        return self.decoder(x) 
 
    # Проверка наличия водяного знака 
    def discriminate(self, x): 
        return self.discriminator(x) 
 
    # Проведение атаки на изображения 
    def attack(self, batch): 
        noise_batch = torch.ones(batch.size()).to(device) 
        for i in range(batch.size()[0]): 
            noise_batch[i] = self.attack_class.random_attack(batch[i]) 
        return noise_batch 
 
    # Вычисление ошибки модели 
    def compute_loss( 
        self,  
        cover: torch.Tensor,  
        watermark: torch.Tensor,  
        cover_norm: torch.Tensor  
    ) -> torch.Tensor: 
 
        encode_image = self.encode(cover_norm, watermark, cover) 
        is_watermark = self.discriminate(encode_image)
        encode_loss = self.criterion(cover,encode_image) 
        discriminate_loss = - torch.log(is_watermark + 0.0001).mean() 
 
        noise_image = self.attack(encode_image) 
        decode_image = self.decode(encode_image) 
        not_watermark = self.discriminate(cover) 
 
        decode_loss = self.criterion(watermark,decode_image) 
        discriminate_loss = discriminate_loss - torch.log(1 - not_watermark + 0.0001).mean() 
        loss = self.alfa * encode_loss + self.beta * decode_loss + self.sigma * discriminate_loss 
 
        return loss 


# Load:

PATH = 'savedmodel/model.pth' # веса обученной модели

net = AutoEncoder()
net.load_state_dict(torch.load(PATH)) # подключение весов
net.eval() # для оценки модели на неизвестных данных
# отключает dropuot и batch normalization для согласованного
# поведения на выходе (главное отличие от .train())


# Загружаем картинку-контейнер
# cover_test = Image.open('PetImages/Cat/21.jpg').convert('RGB').resize((128,128))
cover_img = input("Введите путь и имя изображения-контейнера с расширением: ")
cover_test = Image.open(f'{cover_img}').convert('RGB').resize((128,128))

# Загружаем картинку-watermark
# logo_test = Image.open('PetImages/Dog/21.jpg').convert('L').resize((128,128))
logo_img = input("Введите путь и имя изображения-watermark с расширением: ")
logo_test = Image.open(f'{logo_img}').convert('L').resize((128,128))


# Закодируем водяной знак в контейнер:
# конвертируем картинки в тензоры для подачи в нейросеть
trans = transforms.Compose([transforms.ToTensor()]) 
cover_test = trans(cover_test)

logo_test = trans(logo_test)
test2 = net.encode(cover_test.unsqueeze(0), logo_test.unsqueeze(0), cover_test.unsqueeze(0))

# print(test2)
'''
tensor([[[[0.1623, 0.1433, 0.3059,  ..., 0.1352, 0.1177, 0.0794],
          [0.1373, 0.1399, 0.3212,  ..., 0.1556, 0.1012, 0.1102],
          [0.1463, 0.1538, 0.3396,  ..., 0.1605, 0.0952, 0.0868],
          ...,
          [0.1104, 0.0984, 0.1093,  ..., 0.1468, 0.1353, 0.1218],
          [0.1229, 0.0932, 0.1273,  ..., 0.1472, 0.1654, 0.1168],
          [0.1075, 0.1287, 0.1206,  ..., 0.1577, 0.1442, 0.1463]],

         [[0.1559, 0.1747, 0.3343,  ..., 0.1519, 0.1031, 0.1301],
          [0.1678, 0.1858, 0.3340,  ..., 0.1773, 0.0717, 0.1021],
          [0.1797, 0.1862, 0.3379,  ..., 0.1823, 0.0876, 0.1137],
          ...,
          [0.0916, 0.0995, 0.0931,  ..., 0.1870, 0.1320, 0.1389],
          [0.0811, 0.0776, 0.0994,  ..., 0.1764, 0.1315, 0.1552],
          [0.1031, 0.0886, 0.0892,  ..., 0.1692, 0.1447, 0.1606]],

         [[0.1725, 0.1966, 0.3430,  ..., 0.1534, 0.1330, 0.0992],
          [0.1879, 0.1893, 0.3422,  ..., 0.1698, 0.1218, 0.1103],
          [0.2097, 0.2075, 0.3335,  ..., 0.1716, 0.1242, 0.1073],
          ...,
          [0.0948, 0.0965, 0.0921,  ..., 0.1659, 0.1887, 0.1505],
          [0.1127, 0.1268, 0.1292,  ..., 0.2081, 0.2540, 0.1751],
          [0.1134, 0.0872, 0.1055,  ..., 0.1909, 0.1845, 0.1581]]]],
       grad_fn=<ConvolutionBackward0>)
'''

# print(test2.shape)
# torch.Size([1, 3, 128, 128]) # "набор" (тензор) из 1 изобр, 3 канала(цвета), размеры 128х128

imgtest2 = test2[0] # выбираем из "набора" "картинку"-в-тензорном-формате
# print(imgtest2.shape) # torch.Size([3, 128, 128])

img = transforms.ToPILImage()(imgtest2) # переводим в формат изображения
# print(type(img)) # <class 'PIL.Image.Image'>

# Save images with original and another sizes
# img.save(f"img/{cover_img[:-4]}-with-WM-{logo_img[:-4]}.jpg")
img.resize((1024,1024)).save(f"img/{cover_img[:-4]}-with-WM-{logo_img[:-4]}-resize.jpg")


# Проверим дискриминаторов внедрение (пока не особо понятно как это должно работать)
img = trans(img)
# test = net.discriminate(img.unsqueeze(0))
# print(test) # tensor([[0.]], grad_fn=<SigmoidBackward0>) ??


# Извлечём внедрённую картинку
test3 = net.decode(img.unsqueeze(0)) # конвертируем картинки в тензоры для подачи в нейросеть-decoder
# print(test3.shape) # torch.Size([1, 1, 128, 128]) # "набор" (тензор) из 1 изобр, 1 канал(цвет), размеры 128х128

imgtest3 = test3[0] # выбираем из "набора" "картинку"-в-тензорном-формате
# print(imgtest3.shape) # torch.Size([1, 128, 128])

img3 = transforms.ToPILImage()(imgtest3) # переводим в формат изображения

# Save images with original and another sizes
# img3.save(f"img/from-{cover_img[:-4]}-extract-WM-{logo_img[:-4]}.jpg")
img3.resize((1024,1024)).save(f"img/from-{cover_img[:-4]}-extract-WM-{logo_img[:-4]}-resize.jpg")


# Атакуем изображения
testattack = Attack() # создаём экземпляр класса атак для тестов
cover_img = input("Введите путь и имя атакуемого изображения с расширением: ")
cover_test = Image.open(f'{cover_img}').convert('RGB').resize((128,128)) # загружаем тестируемое изображение
trans = transforms.Compose([transforms.ToTensor()]) 
cover_test = trans(cover_test) # конвертируем картинки в тензоры для подачи в нейросеть

test2attack = testattack.salt(cover_test.unsqueeze(0)) # атака соль-перец (шум на изображении)

img4 = test2attack[0]
img4 = transforms.ToPILImage()(img4)
img4.resize((1280,1024)).save(f"img/{cover_img[:-4]}-salt-and-peper-attack.jpg")



cover_test = Image.open(f'{cover_img}').convert('RGB').resize((128,128))
cover_test = trans(cover_test)

test2attack = testattack.jpg(cover_test)# and ".dropout" too: # атака сжатием jpg и удалением частей изображения dropout

img4 = test2attack[0]
img4 = transforms.ToPILImage()(img4)
img4.resize((1280,1024)).save(f"img/{cover_img[:-4]}-jpg-attack.jpg")