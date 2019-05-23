import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from skimage import io
import os
import numpy as np
import csv
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn import cluster
import sys

images_path = sys.argv[1]
test_csv_file = sys.argv[2]
prediction_file_path = sys.argv[3]

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        nFilters = 8
        hidden_size = 16
        output_channels = 3
        self.encoder = nn.Sequential(
            nn.Conv2d(3, nFilters, 4, stride=2, padding=1),
            nn.BatchNorm2d(nFilters),
            nn.ReLU(True),
            nn.Conv2d(nFilters, 2*nFilters, 4, stride=2, padding=1),
            nn.BatchNorm2d(2*nFilters),
            nn.ReLU(True),
            nn.Conv2d(2*nFilters, 4*nFilters, 4, stride=2, padding=1),
            nn.BatchNorm2d(4*nFilters),
            nn.ReLU(True),
            nn.Conv2d(4*nFilters, hidden_size, 4, stride=2, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_size, 4*nFilters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(4*nFilters),
            nn.ReLU(True),
            nn.ConvTranspose2d(4*nFilters, 2*nFilters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(2*nFilters),
            nn.ReLU(True),
            nn.ConvTranspose2d(2*nFilters, nFilters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(nFilters),
            nn.ReLU(True),

            nn.ConvTranspose2d(nFilters, output_channels, kernel_size=4, stride=2, padding=1),
        )
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
model = torch.load('final_model.pkl')
model.eval()

imgs = []
for i in range(1,40001):
    img = io.imread(images_path + ("%06d" % i) +".jpg")
    img_tensor = transforms.ToTensor()(img)
    imgs.append(np.array(img_tensor))
imgs = np.array(imgs)
imgs_t = torch.FloatTensor(imgs)

code = model.encoder(imgs_t)
embeddings = TSNE(n_jobs=4).fit_transform(np.array(code.data).reshape((40000,-1)))
kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(embeddings)
labels = []
with open(test_csv_file, newline='') as csvfile:
    rows = csv.reader(csvfile)
    
  # 以迴圈輸出每一列
    for n, row in enumerate(rows):
        if n != 0:
            r1 = int(row[1]) - 1
            r2 = int(row[2]) - 1
            if kmeans.labels_[r1] == kmeans.labels_[r2]:
                labels.append(1)
            else:
                labels.append(0)
                
with open(prediction_file_path, 'w', newline='') as csvfile:
    # 建立 CSV 檔寫入器
    writer = csv.writer(csvfile)

    # 寫入一列資料
    writer.writerow(['id', 'label'])
    for _id, i in enumerate(labels):
        writer.writerow([_id, i])