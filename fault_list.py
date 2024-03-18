# from CMD give prob, N, arg_seed
import sys
import numpy as np
import random
import torch
from model.util.MaskGenerator import MaskGenerator, FaultType

# remove "ResNet." from modules name if convenient
# ResNet.ContainerLayer is the container layer inside the ResNet module
# {ModuleName}{i} is the i-th {ModuleName} module instanciated inside the parent module
# modules = [
#     ("ResNet.ContainerLayer", [16, 32, 32]),
#     ("ResNet.Sequential0.BasicBlock0.ContainerLayer0", [16, 32, 32]),
#     ("ResNet.Sequential0.BasicBlock0.ContainerLayer1", [16, 32, 32]),
#     ("ResNet.Sequential0.BasicBlock1.ContainerLayer0", [16, 32, 32]),
#     ("ResNet.Sequential0.BasicBlock1.ContainerLayer1", [16, 32, 32]),
#     ("ResNet.Sequential0.BasicBlock2.ContainerLayer0", [16, 32, 32]),
#     ("ResNet.Sequential0.BasicBlock2.ContainerLayer1", [16, 32, 32]),
#     ("ResNet.Sequential1.BasicBlock0.ContainerLayer0", [32, 16, 16]),
#     ("ResNet.Sequential1.BasicBlock0.ContainerLayer1", [32, 16, 16]),
#     ("ResNet.Sequential1.BasicBlock1.ContainerLayer0", [32, 16, 16]),
#     ("ResNet.Sequential1.BasicBlock1.ContainerLayer1", [32, 16, 16]),
#     ("ResNet.Sequential1.BasicBlock2.ContainerLayer0", [32, 16, 16]),
#     ("ResNet.Sequential1.BasicBlock2.ContainerLayer1", [32, 16, 16]),
#     ("ResNet.Sequential2.BasicBlock0.ContainerLayer0", [64, 8, 8]),
#     ("ResNet.Sequential2.BasicBlock0.ContainerLayer1", [64, 8, 8]),
#     ("ResNet.Sequential2.BasicBlock1.ContainerLayer0", [64, 8, 8]),
#     ("ResNet.Sequential2.BasicBlock1.ContainerLayer1", [64, 8, 8]),
#     ("ResNet.Sequential2.BasicBlock2.ContainerLayer0", [64, 8, 8]),
#     ("ResNet.Sequential2.BasicBlock2.ContainerLayer1", [64, 8, 8])
# ]
modules = [
    ("cont1.injection", [16, 32, 32]),
    ("layer1.0.cont1.injection", [16, 32, 32]),
    ("layer1.0.cont2.injection", [16, 32, 32]),
    ("layer1.1.cont1.injection", [16, 32, 32]),
    ("layer1.1.cont2.injection", [16, 32, 32]),
    ("layer1.2.cont1.injection", [16, 32, 32]),
    ("layer1.2.cont2.injection", [16, 32, 32]),
    ("layer2.0.cont1.injection", [32, 16, 16]),
    ("layer2.0.cont2.injection", [32, 16, 16]),
    ("layer2.1.cont1.injection", [32, 16, 16]),
    ("layer2.1.cont2.injection", [32, 16, 16]),
    ("layer2.2.cont1.injection", [32, 16, 16]),
    ("layer2.2.cont2.injection", [32, 16, 16]),
    ("layer3.0.cont1.injection", [64, 8, 8]),
    ("layer3.0.cont2.injection", [64, 8, 8]),
    ("layer3.1.cont1.injection", [64, 8, 8]),
    ("layer3.1.cont2.injection", [64, 8, 8]),
    ("layer3.2.cont1.injection", [64, 8, 8]),
    ("layer3.2.cont2.injection", [64, 8, 8])
]


device = torch.device('cuda')
n = 1000
prob = 0.025
arg_seed = 0
if len(sys.argv) == 2:
    prob = float(sys.argv[1])
elif len(sys.argv) == 3:
    prob = float(sys.argv[1])
    n = int(sys.argv[2])
elif len(sys.argv) == 4:
    prob = float(sys.argv[1])
    n = int(sys.argv[2])
    seed = int(sys.argv[3])
random.seed(arg_seed)
torch.rand(arg_seed)


def generate_fault_list(n, prob, device):
    res = []
    gen = MaskGenerator(device)
    for _ in range(n):
        layer_name, input_shape = random.choice(modules)
        fault_type = FaultType.rand()
        faulty_idx = random.randint(0, input_shape[fault_type.value-1]-1)
        mask = gen.generateMask(input_shape[0], input_shape[1], input_shape[2], prob, [faulty_idx], fault_type)
        res.append((layer_name, fault_type, mask))
    
    return res

def save_faults(faults):
    to_save = [{'name': fault[0], 'type': fault[1], 'mask': fault[2].numpy(force=True)} for fault in faults]
    np.savez('faults.npz', data=to_save)

def load_faults(file_name='faults.npz'):
    loaded_data = np.load(file_name, allow_pickle=True)
    data = loaded_data['data']
    faults = []
    for item in data:
        name = item['name']
        type = item['type']
        mask = torch.from_numpy(item['mask'])
        if device == torch.device('cuda'):
            mask = mask.cuda()
        fault = (name, type, mask)
        faults.append(fault)
    return faults


# faults = generate_fault_list(n, prob, device)
# save_faults(faults)
# faults = load_faults()
# print(faults)