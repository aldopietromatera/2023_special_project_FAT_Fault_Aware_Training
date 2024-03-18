 
import copy
import torch
import sys

from model.resnet import resnet20
from model.utils import load_CIFAR10_datasets, load_from_dict, get_module_by_name
from utils import test_accuracy
from fault_list import load_faults

device = torch.device('cuda')

# Load the network
prob = 0.025
arg_seed = 0

if len(sys.argv) == 2:
    prob = float(sys.argv[1])
elif len(sys.argv) == 3:
    prob = float(sys.argv[1])
    arg_seed = float(sys.argv[2])

model = resnet20(prob, arg_seed)
model.to(device)
training_file = 'p0025__20230527__h10_22__both.pt'
# training_file = 'ResNet20.th'
# training_file = 'p0025__20230522__h18_52__bwd.pt'
load_from_dict(model, device, 'model/pretrained_models/'+training_file, load_from_golden=False)
print(f'Training file: {training_file}')
print(f'Faults prob = {prob*100:.1f}%')

# Load the test set
train_loader, val_loader, test_loader = load_CIFAR10_datasets(test_batch_size=128,
                                                              train_split=.9)

# [(id, layer (nome o idx), row/col/ch, (idx, val)/mask)]
# getModuleName
# formato npz       np.savez_compressed
# iniettare uno per volta
# n ~ 1000

with torch.no_grad():
    # Test the accuracy
    test_accuracy(model, test_loader, device)

    # Inject
    fault_index = (1, 7, 0, 2)
    layer = model.layer3[0].cont1.conv

    golden = copy.deepcopy(layer.weight[fault_index])
    layer.weight[fault_index] = 1e37
    test_accuracy(model, test_loader, device)
    layer.weight[fault_index] = golden
    
    # for sui faults
    #   inizezione fault
    #   calcolo accuracy su test set
    #   pulizia fault

    #load faults.npz
    print("\nInjection from fault list just started\n")
    faultlist = load_faults()

    for name, type, mask in faultlist:
        mod = get_module_by_name(model, name)
        mod.clear_mask()
        mod.set_manualMask(True)

    for name, type, mask in faultlist:
        mod = get_module_by_name(model, name)
        mod.apply_mask(mask, type)
        mod
        test_accuracy(model, test_loader, device)
        mod.clear_mask()
       