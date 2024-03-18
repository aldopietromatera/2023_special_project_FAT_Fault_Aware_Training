 
# https://github.com/akamaster/pytorch_resnet_cifar10
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import CrossEntropyLoss

from model.layer.ContainerLayer import ContainerLayer

# TODO: replace ReLU with hardtanh (https://pytorch.org/docs/stable/generated/torch.nn.Hardtanh.html)

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, prob, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        #self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.cont1 = ContainerLayer(prob, in_planes, planes, kernel_size=3, stride=stride, padding=1, bias = False)

        #self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.cont2 = ContainerLayer(prob, planes, planes, kernel_size=3, stride=1, padding=1, bias = False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )
    
    def init_mask(self):
        self.cont1.init_mask()
        self.cont2.init_mask()

    def forward(self, x):
        out = self.cont1(x)
        #out = self.bn1(out)
        #out = self.injection(out)

        out = F.relu(out)

        out = self.cont2(out)
        #out = self.bn2(out)
        #out = self.injection(out)

        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

    #TODO: define the backward and try to implement the FI during training



class ResNet(nn.Module):
    def __init__(self, block, num_blocks, prob=0, arg_seed=0, num_classes=10):
        super(ResNet, self).__init__()
        torch.manual_seed(arg_seed)
        self.in_planes = 16

        #self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(16)
        #self.cont1 = ContainerLayer(3, 16, prob, stride=1)
        self.cont1 = ContainerLayer(prob, 3, 16, kernel_size=3, stride=1, padding=1, bias = False)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], prob, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], prob,  stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], prob, stride=2)
        self.linear = nn.Linear(64, num_classes)

        #self.injection = InjectionLayer(prob)

        # Append all layers to a list
        self.layers = None
        self.generate_layer_list()

        self.apply(_weights_init)
    
    def init_mask(self):
        self.cont1.init_mask()
        for x in self.layer1.children():
            x.init_mask()
        for x in self.layer2.children():
            x.init_mask()
        for x in self.layer3.children():
            x.init_mask()

    def generate_layer_list(self):
        self.layers = [self.cont1,
                       #self.bn1,
                       #self.injection,
                       lambda x: F.relu(x),
                       self.layer1,
                       self.layer2,
                       self.layer3,
                       lambda out: F.avg_pool2d(out, out.size()[3]),
                       lambda out: out.view(out.size(0), -1),
                       self.linear]

    def _make_layer(self, block, planes, num_blocks, prob, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, prob, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.cont1(x)
        #out = self.injection(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def resnet20(prob=0, arg_seed=0):
    return ResNet(BasicBlock, [3, 3, 3], prob, arg_seed)


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
