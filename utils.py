from tqdm import tqdm

import torch


def test_golden(network, loader, device):

    network.eval()

    correct_predictions = 0
    total_prediction = 0

    pbar = tqdm(loader)

    golden_logit = list()

    for batch in pbar:
        data, label = batch
        data, label = data.to(device), label.to(device)

        logit = network.to(device)(data)

        golden_logit.append(logit)

        pred_label = torch.argmax(logit, dim=1)

        correct_predictions += sum(pred_label == label)
        total_prediction += loader.batch_size
        accuracy = 100 * correct_predictions / total_prediction

        pbar.set_postfix({'Accuracy': f'{accuracy:.2f}%'})

    return golden_logit



def test_faulty(network, loader, device, golden_logit):
    network.eval()

    different_predictions = 0
    total_prediction = 0

    pbar = tqdm(loader)

    for batch_id, batch in enumerate(pbar):
        data, _ = batch
        data = data.to(device)

        logit = network.to(device)(data)

        pred_label = torch.argmax(logit, dim=1)
        golden_label = torch.argmax(golden_logit[batch_id], dim=1)

        different_predictions += sum(pred_label != golden_label)
        total_prediction += loader.batch_size
        different = 100 * different_predictions / total_prediction

        pbar.set_postfix({'Different': f'{different:.2f}%'})


def test_accuracy(network, loader, device):

    network.eval()

    correct_predictions = 0
    total_prediction = 0
    accuracy = 0

    pbar = tqdm(loader)

    for batch in pbar:
        data, label = batch
        data, label = data.to(device), label.to(device)

        logit = network.to(device)(data)

        pred_label = torch.argmax(logit, dim=1)

        correct_predictions += sum(pred_label == label)
        total_prediction += loader.batch_size
        accuracy = 100 * correct_predictions / total_prediction

        pbar.set_postfix({'Accuracy': f'{accuracy:.2f}%'})


def get_module_name(module):
    if hasattr(module, '__name__'):
        return module.__name__
    elif hasattr(module, '__class__'):
        return module.__class__.__name__
    else:
        return str(module)
    
def get_nested_module_names(module, parent_name=''):
    module_name = get_module_name(module)
    full_name = parent_name + '.' + module_name if parent_name else module_name

    nested_names = []
    for name, submodule in module.named_children():
        nested_names.extend(get_nested_module_names(submodule, full_name))

    if not nested_names:
        return [full_name]
    else:
        return nested_names