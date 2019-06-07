import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from torchvision import transforms, datasets
import sys

training_filepath = sys.argv[1]
modelpath = "best_model.pth"

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

def readfile(path):
    x_train = []
    x_label = []
    val_data = []
    val_label = []
    
    raw_train = np.genfromtxt(path, delimiter=',', dtype=str, skip_header=1)
    for i in range(len(raw_train)):
        tmp = np.array(raw_train[i, 1].split(' ')).reshape(1, 48, 48)
        if (i % 10 == 0):
            val_data.append(tmp)
            val_label.append(raw_train[i][0])
        else:
            x_train.append(tmp)
            x_train.append(np.flip(tmp, axis=2))    # simple example of data augmentation
            x_label.append(raw_train[i][0])
            x_label.append(raw_train[i][0])

x_train = np.array(x_train, dtype=float) / 255.0
val_data = np.array(val_data, dtype=float) / 255.0
x_label = np.array(x_label, dtype=int)
val_label = np.array(val_label, dtype=int)
x_train = torch.FloatTensor(x_train)
val_data = torch.FloatTensor(val_data)
x_label = torch.LongTensor(x_label)
val_label = torch.LongTensor(val_label)

return x_train, x_label, val_data, val_label

x_train, x_label, val_data, val_label = readfile(training_filepath)

transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor()
                                ])

train_set = TensorDataset(x_train, x_label)
val_set = TensorDataset(val_data, val_label)

batch_size = 256
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8)

def train(train_loader, model, criterion, optimizer, epoch):
    print_freq = 10
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)
        input = input.float()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        
        output = model(input_var)
        loss = criterion(output, target_var)
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                                                  epoch, i, len(train_loader), batch_time=batch_time,
                                                                  data_time=data_time, loss=losses, top1=top1, top5=top5))
def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    model.eval()
    
    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        input = input.float()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        
        output = model(input_var)
        loss = criterion(output, target_var)
        
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1, input.size(0))
        top5.update(prec5, input.size(0))
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                                                  i, len(val_loader), batch_time=batch_time, loss=losses,
                                                                  top1=top1, top5=top5))

print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
      .format(top1=top1, top5=top5))

return top1.avg

class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(model, is_best, filename):
    if is_best:
        model.half()
        torch.save(model.state_dict(), filename)

criterion = nn.CrossEntropyLoss()
best_prec1 = 0
lr = 0.1
momentum = 0.9
weight_decay = 1e-4
optimizer = torch.optim.SGD(model.parameters(), lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
model.cuda()
for epoch in range(120):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    train(train_loader, model, criterion, optimizer, epoch)

prec1 = validate(val_loader, model, criterion)

is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)
    save_checkpoint(model, is_best, modelpath)
