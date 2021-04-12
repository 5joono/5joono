import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

torch.cuda.is_available()
cuda = torch.device('cuda')

class bottle(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(bottle, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        if (in_channels != out_channels):
            self.channelfix = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.channelfix = nn.Identity()
            
    def forward(self, x0):
        x = x0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x0 = self.channelfix(x0)
        x = x + x0
        x = self.relu3(x)
        return x

class bottleTrans(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(bottleTrans, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.trans = MHSA(mid_channels, mid_channels)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        if (in_channels != out_channels):
            self.channelfix = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.channelfix = nn.Identity()
            
    def forward(self, x0):
        x = x0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.trans(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x0 = self.channelfix(x0)
        x = x + x0
        x = self.relu3(x)
        return x

class MHSA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MHSA, self).__init__()
        self.layer1 = nn.Identity()

    def forward(self, x):
        x = self.layer1(x)
        return x
        
class botnet(nn.Module):
    def __init__(self):
        super(botnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.multibottle1 = nn.Sequential(
            bottle(64, 64, 256),
            bottle(256, 64, 256),
            bottle(256, 64, 256)
        )
        self.multibottle2 = nn.Sequential(
            bottle(256, 128, 512),
            bottle(512, 128, 512),
            bottle(512, 128, 512),
            bottle(512, 128, 512)
        )
        self.multibottle3 = nn.Sequential(
            bottle(512, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 64, 1024),
            bottle(1024, 64, 1024)
        )
        self.multibottle4 = nn.Sequential(
            bottleTrans(1024, 512, 2048),
            bottleTrans(2048, 512, 2048),
            bottleTrans(2048, 512, 2048)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 10)
    
    def forward(self,x0):
        x = self.conv1(x0)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)
        
        x = self.multibottle1(x)
        x = self.multibottle2(x)
        x = self.multibottle3(x)
        x = self.multibottle4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

net = botnet().cuda()

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].cuda(), data[1].cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].cuda(), data[1].cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].cuda(), data[1].cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(10):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
