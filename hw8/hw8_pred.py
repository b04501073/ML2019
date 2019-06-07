import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import numpy as np
import csv
import sys

# test_filepath = "data/test.csv"
# output_path = "predict_y"
test_filepath = sys.argv[1]
output_path = sys.argv[2]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
    
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(  1,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64,  64, 1),
            conv_dw( 64,  64, 1),
            conv_dw( 64,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 128, 1),
            nn.AvgPool2d(2),
        )
        self.fc = nn.Linear(128, 7)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 128)
        x = self.fc(x)
        return x
model = Net()

model.load_state_dict(torch.load("model_compressed.pth", map_location='cpu'))
model.eval()

def readfile(path):
    print("Reading File...")
    x_test = []

    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        x_test.append(tmp)

    x_test = np.array(x_test, dtype=float) / 255.0
    x_test = torch.FloatTensor(x_test)

    return x_test

x_test = readfile(test_filepath)
output = model(x_test)
prd_class = np.array(output.data)
prd_class = prd_class.argmax(axis=-1)

with open(output_path, 'w') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(['id', 'label'])
    for i in range(len(prd_class)):
        csv_writer.writerow([i]+[prd_class[i]])
