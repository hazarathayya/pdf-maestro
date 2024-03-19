import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder


data_dir = './data/cifar10'
classes = os.listdir(data_dir + '/train')
print(classes)
files = os.listdir(data_dir + '/train/horse')
len(files)

files[:10]

files = os.listdir(data_dir + "/test/frog")
len(files), files[:10]

dataset = ImageFolder(data_dir+'/train', transforms.ToTensor())
dataset

dataset.classes

img, label = dataset[0]
print(img.shape, label)
img

# Commented out IPython magic to ensure Python compatibility.
import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

def show_example(img, label):
  print("label: ", dataset.classes[label])
  plt.imshow(img.permute(1, 2, 0))

show_example(*dataset[23456])

dataset.classes


random_seed = 42
torch.manual_seed(random_seed)

val_size = 5000
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
len(train_ds), len(val_ds)

jovian.log_dataset(dataset_url=Download_URL, val_size=val_size, random_seed=random_seed)

from torch.utils.data.dataloader import DataLoader

batch_size= 128

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size*2, num_workers=2, pin_memory=True)

from torchvision.utils import make_grid

def show_batch(dl):
  for imgs, labels in dl:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks([]);
    ax.set_yticks([]);
    ax.imshow(make_grid(imgs, nrow=16).permute(1, 2, 0))
    break

show_batch(train_dl)

jovian.commit(project=project_name, environment=None)

def apply_kernel(image, kernel):
  ri, ci = image.shape
  rk, ck = kernel.shape
  ro, co = ri-rk+1, ci-ck+1
  out = torch.zeros(ro, co)
  for i in range(ro):
    for j in range(co):
      out[i][j] = torch.sum(image[i:i+rk, j:j+ck]*kernel)
  return out

sample_image = torch.tensor([
    [3, 3, 2, 1, 0],
    [0, 0, 1, 3, 1],
    [3, 1, 2, 2, 3],
    [2, 0, 0, 2, 2],
    [2, 0, 0, 0, 1]
], dtype=torch.float32)

sample_kernel = torch.tensor([
    [0, 1, 2],
    [2, 2, 0],
    [0, 1, 2]
], dtype=torch.float32)

apply_kernel(sample_image, sample_kernel)

torch.zeros([5, 6])

import torch.nn as nn
import torch.nn.functional as F

simple_model = nn.Sequential(
    nn.Conv2d(3, 12, stride=1, kernel_size=3, padding=1),
    nn.MaxPool2d(2, 2)
)

for images, labels in train_dl:
  print('images.shape', images.shape)
  out = simple_model(images)
  print('out.shape', out.shape)
  break

class ImageClassificationBase(nn.Module):
  def training_step(self, batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out, labels)
    return loss

  def validation_step(self, batch):
    images, labels = batch
    out = self(images)
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return {'val_loss':loss, 'val_acc':acc}

  def validation_epoch_end(self, outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_acc = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_acc).mean()
    return {'val_loss':epoch_loss, 'val_acc':epoch_acc}

  def epoch_end(self, epoch, result):
    print(f"Epoch : {epoch+1}, train_loss:{result['train_loss']} loss:{result['val_loss']}, acc:{result['val_acc']}")

def accuracy(out, label):
  _, preds = torch.max(out, dim=1)
  return (torch.sum(preds==label))/len(preds)

class Cifar10cnnModel(ImageClassificationBase):
  def __init__(self):
    super(Cifar10cnnModel, self).__init__()
    self.network = nn.Sequential(
        nn.Conv2d(3, 12, stride=1, kernel_size=3, padding=1), # 3 x 32 x 32  -> 12 x 32 x 32
        nn.ReLU(),
        nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1), # 16 x 32 x 32
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 16 x 16 x 16

        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32 x 16 x 16
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64 x 16 x 16
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 64 x 8 x 8

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128 x 8 x 8
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 256 x 8 x 8
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 256 x 4 x 4

        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 512 x 4 x 4
        nn.ReLU(),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # 1024 x 4 x 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 1024 x 2 x 2

        nn.Flatten(), # 1024
        nn.Dropout(0.2),
        nn.Linear(4096, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    )

  def forward(self, x):
    x = self.network(x)
    return x

1024*4

model = Cifar10cnnModel()
model

for images, labels in train_dl:
    print('images.shape:', images.shape)
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)

device = get_default_device()
device

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device);

# jovian.commit(project=project_name)

@torch.no_grad()
def evaluate(model, val_loader):
  model.eval()
  outputs = [model.validation_step(batch) for batch in val_loader]
  return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.Adam):
  history = []
  optimizer = opt_func(model.parameters(), lr, weight_decay=1e-6)
  for epoch in range(epochs):
    model.train()
    train_losses = []
    for batch in train_loader:
      loss = model.training_step(batch)
      train_losses.append(loss)
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

    result = evaluate(model, val_loader)
    result['train_loss'] = torch.stack(train_losses).mean().item()
    model.epoch_end(epoch, result)
    history.append(result)
  return history

model = to_device(Cifar10cnnModel(), device)

evaluate(model, val_dl)

num_epochs = 10
opt_func = torch.optim.Adam
lr = 0.001

jovian.reset()
jovian.log_hyperparams({
    'num_epochs':num_epochs,
    'opt_func':opt_func.__name__,
    'batch_size':batch_size,
    'lr':lr,
})

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

def plot_accuracies(history):
    accuracies = [x['val_acc'].cpu().detach().numpy() for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs');

plot_accuracies(history)

num_epochs = 10
lr = 0.0001
opt_func = torch.optim.Adam

history1 = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

history = history + history1
plot_accuracies(history)

def plot_losses(history):
  train_losses = [x['train_loss'] for x in history]
  val_losses = [x['val_loss'].cpu().detach().numpy() for x in history]
  plt.plot(train_losses, '-bx', label = 'Training')
  plt.plot(val_losses, '-rx', label = 'validation')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.legend()
  plt.title('Loss vs No of epochs')

plot_losses(history)

jovian.commit(project=project_name)

test_dataset = ImageFolder(data_dir+'/test', transform=transforms.ToTensor())

def predict_image(img, model):
  # convert the image to batch of 1 image
  x = to_device(img.unsqueeze(0), device)
  # obtain the predictions
  y = model(x)
  # pick the index with high probability
  _, preds = torch.max(y, dim=1)
  # Retreive the class label
  return dataset.classes[preds[0].item()]

img, label = test_dataset[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

img, label = test_dataset[1002]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
result

def detect_wrong_preds(val_loader):
  for batch in val_loader:
    imgs, labels = batch
    out = model(imgs)
    _, preds = torch.max(out, dim=1)
    got = preds == labels
    for i in range(len(got)):
      if not got[i]:
        plt.imshow(imgs[i].permute(1, 2, 0).cpu().detach().numpy())
        plt.show()
        print('Label:', dataset.classes[labels[i]], ', predicted:', dataset.classes[preds[i]])
    break

detect_wrong_preds(val_dl)

def detect_wrong_preds(val_loader):
  for batch in val_loader:
    imgs, labels = batch
    out = model(imgs)
    _, preds = torch.max(out, dim=1)
    got = preds == labels
    for i in range(len(got)):
      if not got[i]:
        plt.imshow(imgs[i].permute(1, 2, 0).cpu().detach().numpy())
        plt.show()
        print('Label:', dataset.classes[labels[i]], ', Predicted:', predict_image(imgs[i], model))
    break

got = detect_wrong_preds(val_dl)