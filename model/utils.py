import os
from functools import reduce
from tqdm import tqdm

import torch
from torch.nn import Module
from torch.utils.data import TensorDataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageNet, MNIST

import copy


def get_module_by_name(container_module: Module,
                       module_name: str) -> Module:
    """
    Return the instance of the submodule module_name inside the container_module
    :param container_module: The container module that contains the module_name module
    :param module_name: The name of the module to find
    :return: The instance of the submodule with the specified name
    """

    # To fine the actual layer with nested layers (e.g. inside a convolutional layer inside a Basic Block in a
    # ResNet, first separate the layer names using the '.'
    formatted_names = module_name.split(sep='.')

    # Access the nested layer iteratively using itertools.reduce and getattr
    module = reduce(getattr, formatted_names, container_module)

    return module


def load_ImageNet_validation_set(batch_size,
                                 image_per_class=None,
                                 network=None,
                                 imagenet_folder='~/Datasets/ImageNet'):
    """

    :param batch_size:
    :param image_per_class:
    :param network: Default None. The network used to select the image per class. If not None, select the image_per_class
    that maximize this network accuracy. If not specified, images are selected at random
    :param imagenet_folder:
    :return:
    """

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    transform_validation = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    validation_dataset_folder = 'tmp'
    validation_dataset_path = f'{validation_dataset_folder}/imagenet_{image_per_class}.pt'

    try:
        if image_per_class is None:
            raise FileNotFoundError

        validation_dataset = torch.load(validation_dataset_path)
        print('Resized Imagenet loaded from disk')

    except FileNotFoundError:
        validation_dataset = ImageNet(root=imagenet_folder,
                                      split='val',
                                      transform=transform_validation)

        if image_per_class is not None:
            selected_validation_list = []
            image_class_counter = [0] * 1000

            # First select only correctly classified images
            for validation_image in tqdm(validation_dataset, desc='Resizing Imagenet Dataset', colour='Yellow'):
                if image_class_counter[validation_image[1]] < image_per_class:
                    prediction = network(validation_image[0].cuda().unsqueeze(dim=0)).argmax() if network is not None else validation_image[1]
                    if prediction == validation_image[1]:
                        selected_validation_list.append(validation_image)
                        image_class_counter[validation_image[1]] += 1

            # Then select images to fill up
            for validation_image in tqdm(validation_dataset, desc='Resizing Imagenet Dataset', colour='Yellow'):
                if image_class_counter[validation_image[1]] < image_per_class:
                    selected_validation_list.append(validation_image)
                    image_class_counter[validation_image[1]] += 1
            validation_dataset = selected_validation_list

        os.makedirs(validation_dataset_folder, exist_ok=True)
        torch.save(validation_dataset, validation_dataset_path)

    # DataLoader is used to load the dataset
    # for training
    val_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    print('Dataset loaded')

    return val_loader


def load_MNIST_datasets(train_batch_size=32, test_batch_size=1):

    train_loader = torch.utils.data.DataLoader(
        MNIST('weights/files/', train=True, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Resize((32, 32)),
                  transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=train_batch_size, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        MNIST('weights/files/', train=False, download=True,
              transform=transforms.Compose([
                  transforms.ToTensor(),
                  transforms.Resize((32, 32)),
                  transforms.Normalize(
                      (0.1307,), (0.3081,))
              ])),
        batch_size=test_batch_size, shuffle=True)

    print('Dataset loaded')

    return train_loader, test_loader


def load_CIFAR10_datasets(train_batch_size=32, train_split=0.8, test_batch_size=1, test_image_per_class=None):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),                                       # Crop the image to 32x32
        transforms.RandomHorizontalFlip(),                                          # Data Augmentation
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),                                                  # Crop the image to 32x32
        transforms.ToTensor(),                                                      # Transform from image to pytorch tensor
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),   # Normalize the data (stability for training)
    ])

    train_dataset = CIFAR10('weights/files/',
                            train=True,
                            transform=transform_train,
                            download=True)
    test_dataset = CIFAR10('weights/files/',
                           train=False,
                           transform=transform_test,
                           download=True)

    # If only a number of images is required per class, modify the test set
    if test_image_per_class is not None:
        image_tensors = list()
        label_tensors = list()
        image_class_counter = [0] * 10
        for test_image in test_dataset:
            if image_class_counter[test_image[1]] < test_image_per_class:
                image_tensors.append(test_image[0])
                label_tensors.append(test_image[1])
                image_class_counter[test_image[1]] += 1
        test_dataset = TensorDataset(torch.stack(image_tensors),
                                     torch.tensor(label_tensors))

    # Split the training set into training and validation
    train_split_length = int(len(train_dataset) * train_split)
    val_split_length = len(train_dataset) - train_split_length
    train_subset, val_subset = torch.utils.data.random_split(train_dataset,
                                                             lengths=[train_split_length, val_split_length],
                                                             generator=torch.Generator().manual_seed(1234))
    # DataLoader is used to load the dataset
    # for training
    train_loader = torch.utils.data.DataLoader(dataset=train_subset,
                                               batch_size=train_batch_size,
                                               shuffle=True)

    if train_split < 1:
        val_loader = torch.utils.data.DataLoader(dataset=val_subset,
                                                 batch_size=train_batch_size,
                                                 shuffle=True)
    else:
        val_loader = None

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False)

    print('Dataset loaded')

    return train_loader, val_loader, test_loader

def mapping(state_dict2):

    '''
        Takes in a state_dict from a pretrained model and map the layer after having added the containerLayer.

            Parameters:
                    state_dict2 (collections.OrderedDict): state_dict from a pretrained model

            Returns:
                    state_dict (collections.OrderedDict): state_dict after mapping
    '''
     
    state_dict = copy.deepcopy(state_dict2)

    for key in state_dict2.keys():

        if "conv1" in key:
            val = state_dict.pop(key)
            x = key.replace("conv1", "cont1.conv")
            state_dict[x] = val
        elif "bn1" in key:
            val = state_dict.pop(key)
            x = key.replace("bn1", "cont1.bn")
            state_dict[x] = val

        for i in [1, 2, 3]:
            for k in [0, 1, 2]:
                st = "layer"+str(i)+"."+str(k)+".conv2"
                if st in key:
                    val = state_dict.pop(key)
                    x = key.replace("conv2", "cont2.conv")
                    state_dict[x] = val
                else:
                    st = "layer"+str(i)+"."+str(k)+".bn2"
                    if st in key:
                        val = state_dict.pop(key)
                        x = key.replace("bn2", "cont2.bn")
                        state_dict[x] = val

    return state_dict


def load_from_dict(network, device, path, function=None, load_from_golden = False):

    if '.th' in path:
        state_dict = torch.load(path, map_location=device)['state_dict']
    else:
        state_dict = torch.load(path, map_location=device)

    if load_from_golden:
        state_dict = mapping(state_dict)

    if function is None:
        clean_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
    else:
        clean_state_dict = {key.replace('module.', ''): function(value) if not (('bn' in key) and ('weight' in key)) else value for key, value in state_dict.items()}

    network.load_state_dict(clean_state_dict, strict=False)
