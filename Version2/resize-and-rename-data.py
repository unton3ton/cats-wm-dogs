import PIL
from PIL import Image
import os, sys
import time

path1 = "wm/" 
color_mode = "L"

# path2 = "cover/" 
# color_mode = "RGB" # "L"

def resize(path):
    dirs = os.listdir(path)
    print('before resize ', len(dirs))
    for item in dirs:
        try:
            # print(item)
            with Image.open(fr'{path}{item}') as im:
                resized = im.convert(f'{color_mode}').resize((64,64))
                resized.save(fr'{path}{item}')
                time.sleep(0.0003)
                # print(fr'for {item} have been done')
        except PIL.UnidentifiedImageError:
            print(fr"Confirmed: This image {path}{item} cannot be opened!")
            # os.remove(f'{path}{item}')
        except OSError:
            im = Image.open(fr'{path}{item}').convert(f'{color_mode}').resize((64,64))
            im.save(fr'{path}{item}')
            print(fr"Chanched by hands for {path}{item}")
    dirs = os.listdir(path)
    print('after resize ', len(dirs))


resize(path1)


def test_size(path):
    dirs = os.listdir(path)
    print('before test ', len(dirs))
    for item in dirs:
        try:
            with Image.open(fr'{path}{item}') as im:
                width, height = im.size
                if width == height == 64:
                    pass
                else:
                    print(fr'for {item} not true size')
                time.sleep(0.0003)
        except PIL.UnidentifiedImageError:
            print(fr"Confirmed: This image {item} cannot be opened! We removed it")
            os.remove(f'{path}{item}')
    dirs = os.listdir(path)
    print('after test ', len(dirs))


test_size(path1)


def renameimg(path):
    os.getcwd()
    for i, filename in enumerate(os.listdir(path)):
        try:
            os.rename(path + filename, path + str(i) + ".jpg")
        except FileExistsError:
            pass

# renameimg(path2)
