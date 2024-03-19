import os
from PIL import Image
import pandas as pd

def create_label(path, image, deg):
    label = 0
    img_path = os.path.join(path, f"data/Images/{image}")
    im = Image.open(img_path)
    r_im = im.rotate(deg)
    r_im.save(img_path)
    if deg==90:
        label = 1
    elif deg==180:
        label = 2
    elif deg==270:
        label = 3
    elif deg==0:
        label = 0
    return image, label

def test(path):
    for i in os.listdir(path + '/data/Images'):
        img_path = os.path.join(path, f"data/Images/{i}")
        print(f"path ->{path}\n,i-> {i}\n,img_path -> {img_path}\n")
        im = Image.open(img_path)
        # im.show()
        r_im = im.rotate(270)
        r_im.show()
        r_im.save(img_path)
        print(im)
        break

def create_labels(path):
    data = []
    # print(os.getcwd())
    count = 0
    for i in os.listdir(path + '/data/Images'):
        if count%4==0:
            data.append(create_label(path, i, 0))
        elif count%4==1:
            data.append(create_label(path, i, 90))
        elif count%4==2:
            data.append(create_label(path, i, 180))
        elif count%4==3:
            data.append(create_label(path, i, 270))
        count = count+1
    return data


print(os.getcwd())
# os.chdir('./data/Images')
dir = os.getcwd()
# test(dir)

im_lb = create_labels(dir)
df = pd.DataFrame(im_lb)

os.chdir('./data')
# print(os.getcwd())
df.to_csv('pdfimages.csv')