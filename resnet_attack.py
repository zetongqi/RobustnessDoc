import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim


img_path = "/Users/zetong/Desktop/dog3.jpg"

def processing(img_path):
	img = Image.open(img_path)
	preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),])
	img_tensor = preprocess(img)[None, :, :, :]
	return img_tensor

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.Tensor(mean)
        self.std = torch.Tensor(std)
    def forward(self, x):
        return (x - self.mean.type_as(x)[None,:,None,None]) / self.std.type_as(x)[None,:,None,None]

def predict(model, img_tensor):
	norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	pred = model(norm(img_tensor))
	return pred.max(dim=1)[1].item()


norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
model = models.resnet50(pretrained=True)
model.eval()
img_tensor = processing(img_path)

epsilon = 2./255

delta = torch.zeros_like(img_tensor, requires_grad=True)
opt = optim.SGD([delta], lr=1e-1)

for t in range(30):
    pred = model(norm(img_tensor + delta))
    loss = -nn.CrossEntropyLoss()(pred, torch.LongTensor([207])) + nn.CrossEntropyLoss()(pred, torch.LongTensor([753])) + delta.norm(p=2)
    if t % 5 == 0:
        print(t, loss.item())
    
    opt.zero_grad()
    loss.backward()
    opt.step()
    delta.data.clamp_(-epsilon, epsilon)
    
print("True class probability:", nn.Softmax(dim=1)(pred)[0,207].item())
print(predict(model, img_tensor+delta))
plt.imshow((img_tensor+delta)[0].detach().numpy().transpose(1,2,0))
plt.show()
