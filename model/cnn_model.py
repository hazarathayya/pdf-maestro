import torch
import torch.nn as nn
import torch.nn.functional as F
from dataloader import get_dl


train_dl, val_dl = get_dl()

for images ,labels in train_dl:
    print(images.shape)
    print(labels.shape)
    break

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self, batch):
        images, labels = batch
        out = self(batch)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'loss':loss, 'acc':acc}
    
    def validation_epoch_end(self, outputs):
        batch_loss = [x['loss'] for x in outputs]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_acc = [x['acc'] for x in outputs]
        epoch_acc = torch.stack(batch_acc).mean()
        return {'loss':epoch_loss, 'acc':epoch_acc}

    def epoch_end(self, epoch, result):
        return f"epoch:{epoch+1} epoch_loss:{result['loss']}, epoch_acc:{result['acc']}"
    
def accuracy(out, labels):
    _, preds = torch.max(out, dim=1)
    return torch.sum(preds==labels)/len(out)

class AligncnnModel(ImageClassificationBase):
  def __init__(self):
    super(AligncnnModel, self).__init__()
    self.network = nn.Sequential(
        nn.Conv2d(3, 12, stride=1, kernel_size=3, padding=1), # 3 x 256 x 256  -> 12 x 256 x 256
        nn.ReLU(),
        nn.Conv2d(12, 16, kernel_size=3, stride=1, padding=1), # 16 x 256 x 256
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 16 x 64 x 64

        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # 32 x 16 x 16
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # 64 x 16 x 16
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 64 x 32 x 32

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 128 x 8 x 8
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # 256 x 8 x 8
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 256 x 16 x 16

        nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), # 512 x 4 x 4
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # 1024 x 4 x 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 1024 x 8 x 8

        nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), # 512 x 4 x 4
        nn.ReLU(),
        nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1), # 1024 x 4 x 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 1024 x 8 x 8

        nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1), # 512 x 4 x 4
        nn.ReLU(),
        nn.Conv2d(1024, 2048, kernel_size=3, stride=1, padding=1), # 1024 x 4 x 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 1024 x 8 x 8

        nn.Conv2d(2048, 2048, kernel_size=3, stride=1, padding=1), # 512 x 4 x 4
        nn.ReLU(),
        nn.Conv2d(2048, 4096, kernel_size=3, stride=1, padding=1), # 1024 x 4 x 4
        nn.ReLU(),
        nn.MaxPool2d(2, 2), # 4096 x 8 x 8

        nn.Flatten(), # 4096
        nn.Dropout(0.2),
        nn.Linear(4096, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

  def forward(self, x):
    x = self.network(x)
    return x

model = AligncnnModel()

for images, labels in train_dl:
    print('images.shape:', images.shape)
    print(type(images), type(labels))
    out = model(images)
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break

# def get_default_device():
#     """Pick GPU if available, else CPU"""
#     if torch.cuda.is_available():
#         return torch.device('cuda')
#     else:
#         return torch.device('cpu')

# def to_device(data, device):
#     """Move tensor(s) to chosen device"""
#     if isinstance(data, (list,tuple)):
#         return [to_device(x, device) for x in data]
#     return data.to(device, non_blocking=True)

# class DeviceDataLoader():
#     """Wrap a dataloader to move data to a device"""
#     def __init__(self, dl, device):
#         self.dl = dl
#         self.device = device

#     def __iter__(self):
#         """Yield a batch of data after moving it to device"""
#         for b in self.dl:
#             yield to_device(b, self.device)

#     def __len__(self):
#         """Number of batches"""
#         return len(self.dl)

# device = get_default_device()
# device

# train_dl = DeviceDataLoader(train_dl, device)
# val_dl = DeviceDataLoader(val_dl, device)
# to_device(model, device);

# # jovian.commit(project=project_name)

# @torch.no_grad()
# def evaluate(model, val_loader):
#   model.eval()
#   outputs = [model.validation_step(batch) for batch in val_loader]
#   return model.validation_epoch_end(outputs)


# def fit(epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.Adam):
#   history = []
#   optimizer = opt_func(model.parameters(), lr, weight_decay=1e-6)
#   for epoch in range(epochs):
#     model.train()
#     train_losses = []
#     for batch in train_loader:
#       loss = model.training_step(batch)
#       train_losses.append(loss)
#       loss.backward()
#       optimizer.step()
#       optimizer.zero_grad()

#     result = evaluate(model, val_loader)
#     result['train_loss'] = torch.stack(train_losses).mean().item()
#     model.epoch_end(epoch, result)
#     history.append(result)
#   return history

# model = to_device(AligncnnModel(), device)

# evaluate(model, val_dl)

# num_epochs = 10
# opt_func = torch.optim.Adam
# lr = 0.001

# history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)