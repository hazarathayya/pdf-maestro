import sys
import os
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.io import read_image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import torchvision.transforms as tf

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from image_dataset.torch_datasets import PdfImages

print(os.getcwd())
df = pd.read_csv("./data/pdfimages.csv")
# """
path = os.getcwd()
dataset = PdfImages(df, path, transform=tf.Compose([tf.RandomResizedCrop((256, 256)),
                                                    tf.ToTensor(),
                                                    ]))

train_size = int(len(dataset)*0.8)
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
# print(train_ds[45])
batch_size = 16

def get_dl():
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size*2)
    return train_dl, val_dl

# train_dl, val_dl = get_dl()
# im, l = train_ds[0]
# plt.imshow(im.permute(1, 2, 0))
# plt.show()

def show_batch(dl):
  for imgs, labels in dl:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.imshow(make_grid(imgs, nrow=4).permute(1, 2, 0))
    break
  plt.show()

# To show the batch
# show_batch(train_dl)
  
# for i in range(343, 350):
#   im, l = train_ds[i]
#   print(l)
#   plt.imshow(im.permute(1, 2, 0))
#   plt.show()
#   break   

# """


