import torch.nn as nn

from model.layer.InjectionLayer import InjectionLayer

class ContainerLayer(nn.Module):

    def __init__(self, prob, in_planes, planes, kernel_size, stride, padding, bias):
        super(ContainerLayer, self).__init__()

        self.conv = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(planes)
        self.injection = InjectionLayer(prob)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.injection(out)

        return out
    
    def init_mask(self):
        self.injection.init_mask()

    def apply_mask(self, mask, faultType):
        self.injection.apply_mask(mask, faultType)
