import torchvision.models as models
from torchvision import transforms
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch.optim as optim
import os
import PIL
import scipy.misc
from loss_gradient_attack import (
    img_path,
    get_resnet50,
    processing,
    Normalize,
    predict,
    save_image_from_nparray,
)

# FGSM attack for resnet50
class FGSM:
    def __init__(self, model, epsilon=50 / 255):
        self.model = model
        self.epsilon = epsilon

    def fgsm(self, x, true_label):
        norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        x.requires_grad = True
        pred = self.model(norm(x))
        loss = nn.CrossEntropyLoss()(pred, torch.LongTensor([true_label]))

        model.zero_grad()
        loss.backward()
        sign = x.grad.data.sign()
        delta = self.epsilon * sign[0]
        x_adv = (x[0] + delta).detach().numpy()
        if predict(model, x_adv) != true_label:
            print("attack sucessful!")
            return x_adv
        else:
            print("attack failed!")
            return None


if __name__ == "__main__":
    model = get_resnet50()
    att = FGSM(model)
    x = processing(img_path)
    x_adv = att.fgsm(x, 207)
    print(predict(model, x_adv))
    save_image_from_nparray(x_adv, "retriver_adv_fgsm.png")
