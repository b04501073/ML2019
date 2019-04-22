import numpy as np
import torch 
import torch.nn as nn
import torchvision.transforms as transform
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
from torchvision.models import resnet50
import torch.utils.model_zoo as model_zoo
import requests
import os
from io import BytesIO
import sys

model = resnet50(pretrained = False)
model.load_state_dict(torch.load('fgsm_model.pth'))
model.eval()

label_idx = []
f_file = "hw5_data/labels.csv"
with open(f_file) as file:
    for line_id, line in enumerate(file):
        if line_id != 0:
            datas = line.split(',')
            label = datas[3]
            label_idx.append(int(label))
label_idx = np.array(label_idx)

def threshhold(image, idx, trans_value):
    for i in range(len(image[0])):
        for j in range(len(image[0][i])):
            for k in range(len(image[0][i][j])):
                if idx[0][i][j][k] == 1:
                    image[0][i][j][k] = trans_value * image[0][i][j][k].sign_()
    return image

def clip(image, minvalue, maxvalue):
    for i in range(len(image[0])):
        for j in range(len(image[0][i])):
            for k in range(len(image[0][i][j])):
                if image[0][i][j][k] < minvalue:
                    image[0][i][j][k] = minvalue
                elif image[0][i][j][k] > maxvalue:
                    image[0][i][j][k] = maxvalue
    return image

img_file_path = sys.argv[1]
output_file = sys.argv[2]
criterion = nn.CrossEntropyLoss()
epsilon = 0.007


for i in range(200):
    
    img = Image.open(img_file_path + ("%03d" % i) + ".png")
    trans = transform.Compose([transform.ToTensor()])
    image = trans(img)
    image = image.unsqueeze(0)

    image.requires_grad = True
    zero_gradients(image)

    output = model(image)
    
    rank = output.argsort(-1)[0]
    
    if label_idx[i] != rank[-1]:
        ori_label = label_idx[i]
        target_label = rank[-1]
    else:
        ori_label = rank[-1]
        target_label = rank[-2]
        
    target_label = [target_label]
    target_label = torch.tensor(target_label)
    
    ori_label = [ori_label]
    ori_label = torch.tensor(ori_label)
#     print(ori_label, target_label)

    loss1 = criterion(output, target_label)
    loss2 = criterion(output, ori_label)
    
    loss = loss2 - loss1
    loss.backward()
    
    deltaimage = image.grad.sign_()
    
#     deltaimage = image.grad.data
#     idx = abs(deltaimage) > 0.01
#     deltaimage = threshhold(deltaimage, idx, 0.01)
    
    
    image = image + deltaimage * epsilon
    
#     deltaimage = image.grad.sign_()
#     image = image + epsilon * deltaimage
    
#     image = image + epsilon * image.grad.sign_()
    image = clip(image, 0, 1)
    image_ad = transform.ToPILImage()(image[0])
    image_ad.save(output_file + ("%03d" % i) + ".png")
    img.close()