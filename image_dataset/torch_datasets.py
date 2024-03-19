import os
import torch 
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class PdfImages(Dataset):
    def __init__(self, df, path, transform=None):
        self.df = df
        self.path = path
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.path, f"data/Images/{self.df.iloc[idx, 1]}")
        image = read_image(image_path)
        label = self.df.iloc[idx, 2]

        if self.transform:
            image = self.transform(image)
        
        return image, label


# print(os.getcwd())
# df = pd.read_csv("./data/pdfimages.csv")
# train_ds = PdfImages(df)
# print(train_ds[450])