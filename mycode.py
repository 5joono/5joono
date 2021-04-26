import torch
from torch import einsum, nn, optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from torchsummary import summary

import matplotlib.pyplot as plt
import numpy as np

from einops import rearrange

import json
import os

torch.cuda.is_available()
cuda = torch.device('cuda')

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim=dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, nh, l, _ = x.shape
    col_pad = torch.zeros((b, nh, l, 1)).cuda()
    x = torch.cat((x, col_pad), dim=3)
    flat_x = rearrange(x, 'b nh l c -> b nh (l c)')
    flat_pad = torch.zeros((b, nh, l-1)).cuda()
    flat_x = torch.cat((flat_x, flat_pad), dim=2)
    final_x = torch.reshape(flat_x, (b, nh, l+1, 2*l-1))
    return final_x[:,:,:l,(l-1):]

def relative_logits_1d(q, rel_k):
    b, n, h, w, _ = q.shape
    logits = einsum('b n h w d, r d -> b n h w r', q, rel_k)
    logits = rearrange(logits, 'b n h w r -> b (n h) w r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, n, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

class AbsPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)
        
    def forward(self, q):
        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, 'h w d -> (h w) d')
        logits = einsum('b n x d, y d -> b n x y', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(self, height, width, dim_head):
        super().__init__()
        scale = dim_head ** -0.5
        self.height = height
        self.width = width
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h = self.height
        w = self.width

        q = rearrange(q, 'b n (h w) d -> b n h w d', h=h)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b n h H w W-> b n (h w) (H W)')

        q = rearrange(q, 'b n h w d -> b n w h d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b n w W h H -> b n (h w) (H W)')
        return rel_logits_w + rel_logits_h        
        
class MHSA(nn.Module):
    def __init__(self, in_channels, fmap_size, heads=4, dim_head=128, rel_pos_emb=False):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        out_channels = heads * dim_head
        height, width = fmap_size

        self.to_qkv = nn.Conv2d(in_channels, out_channels*3, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        if rel_pos_emb:
            self.pos_emb = RelPosEmb(height, width, dim_head)
        else:
            self.pos_emb = AbsPosEmb(height, width, dim_head)

    def forward(self, featuremap):
        heads = self.heads
        b, c, h, w = featuremap.shape
        q, k, v = self.to_qkv(featuremap).chunk(3, dim=1)
        q, k, v = map(lambda x: rearrange(x, 'b (n d) h w -> b n (h w) d', n=heads), (q, k, v))

        q *= self.scale

        logits = einsum('b n x d, b n y d -> b n x y', q, k)
        logits += self.pos_emb(q)

        weights = self.softmax(logits)
        attn_out = einsum('b n x y, b n y d -> b n x d', weights, v)
        attn_out = rearrange(attn_out, 'b n (h w) d -> b (n d) h w', h=h)

        return attn_out
    
class bottle(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride=1, attention=False):
        super().__init__()
        self.attention = attention
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        
        self.mhsa = MHSA(mid_channels, (14,14), rel_pos_emb=True)
            
        if (in_channels != out_channels):
            self.channelfix = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            self.avgpool = nn.AvgPool2d(2)
        else:
            self.channelfix = nn.Identity()
            self.avgpool = nn.Identity()
            
    def forward(self, x0):
        x = x0
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        if(self.attention):
            x = self.mhsa(x)
            # x = self.avgpool(x)
        else:
            x = self.conv2(x)
        
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        
        x0 = self.channelfix(x0)
        x = x + x0
        x = self.relu3(x)
        return x

class botnet(nn.Module):
    def __init__(self):
        super(botnet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.multibottle1 = nn.Sequential(
            bottle(64, 64, 256),
            bottle(256, 64, 256),
            bottle(256, 64, 256)
        )
        self.multibottle2 = nn.Sequential(
            bottle(256, 128, 512, 2),
            bottle(512, 128, 512),
            bottle(512, 128, 512),
            bottle(512, 128, 512)
        )
        self.multibottle3 = nn.Sequential(
            bottle(512, 256, 1024, 2),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024),
            bottle(1024, 256, 1024)
        )
        self.multibottle4 = nn.Sequential(
            bottle(1024, 512, 2048, attention=True),
            bottle(2048, 512, 2048, attention=True),
            bottle(2048, 512, 2048, attention=True)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, 1000)
    
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
param = list(net.parameters())
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.ImageNet('./imagenet', split='train', download=None, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.ImageNet('./imagenet', split='val', download=None, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

criterion = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adam(net.parameters(),lr=0.00001)

idx2label = []
cls2label = {}
with open("./imagenet_class_index.json", "r") as read_file:
    class_idx = json.load(read_file)
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}

for epoch in range(3):  # loop over the dataset multiple times
    running_loss = 0.0

    if(epoch>0):
        net = botnet()
        net.load_state_dict(torch.load(save_path))
        net.to(device)

    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if(loss.item() > 1000):
            print(loss.item())
            for param in net.parameters():
                print(param.data)
        # print statistics
        running_loss += loss.item()
        if i % 50 == 49:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
            running_loss = 0.0
    save_path="./my_result.pth"
    torch.save(net.state_dict(), save_path)
        #print("\n")

print('Finished Training')

class_correct = list(0. for i in range(1000))
class_total = list(0. for i in range(1000))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(16):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

accuracy_sum=0
for i in range(1000):
    temp = 100 * class_correct[i] / class_total[i]
    print('Accuracy of %5s : %2d %%' % (
        idx2label[i], temp))
    accuracy_sum+=temp
print('Accuracy average: ', accuracy_sum/1000)
