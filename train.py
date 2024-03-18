import os
import numpy as np
from tqdm import tqdm

import torch
from torch.nn import Conv2d

from torch.nn import CrossEntropyLoss

from model.resnet import resnet20
from model.utils import load_CIFAR10_datasets
from utils import test_accuracy

import csv
import datetime # used for creating unique training files names

device = torch.device('cuda')

# Load the network
# prob is the probability that a neuron is affected by multiplicative noise
prob = 0.025
notes_filename="both"  #it allows tu add some notes at the end of the filename.
model = resnet20(prob=prob)
model.to(device)

# Load the test set
train_loader, val_loader, test_loader = load_CIFAR10_datasets(test_batch_size=128,
                                                              train_split=.9,)


for module in model.modules():
    if isinstance(module, Conv2d):
        if module.weight.requires_grad:
            torch.nn.init.kaiming_normal_(module.weight)

# Train
optimizer = torch.optim.SGD(model.parameters(), lr=1e-1, momentum=.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                    milestones=[100, 150])
loss_fn = CrossEntropyLoss()
all_epoch = 200
prev_acc = 0
counter = 0

# Opening csv file to record #epoch and accuracy
header = ['epoch', 'accuracy']

def unique_name(notes):
    now = datetime.datetime.now()
    # ret = str(now.date()).replace('-', '') + '__h' + str(now.time())[:5].replace(':','_') + '__p'+ format(int(1000*prob), '04')
    ret = 'p'+ format(int(1000*prob), '04') + "__"+ str(now.date()).replace('-', '') + '__h' + str(now.time())[:5].replace(':','_')+"__"+notes
    return ret

filename = unique_name(notes=notes_filename)
with open('csv/'+filename+'.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)

    for current_epoch in range(all_epoch):
        model.init_mask()
        model.train()
        pbar = tqdm(train_loader,
                    desc=f'Epoch {current_epoch}')

        all_correct_num = 0
        all_sample_num = 0

        for idx, (train_x, train_label) in enumerate(pbar):
            train_x = train_x.to(device)
            train_label = train_label.to(device)
            optimizer.zero_grad()
            predict_y = model(train_x.float())

            predict_label = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_label == train_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
            acc = all_correct_num / all_sample_num
            pbar.set_postfix({'Accuracy': acc})

            loss = loss_fn(predict_y, train_label.long())
            loss.backward()
            optimizer.step()

        lr_scheduler.step()

        all_correct_num = 0
        all_sample_num = 0
        model.eval()

        for idx, (test_x, test_label) in enumerate(val_loader):
            test_x = test_x.to(device)
            test_label = test_label.to(device)
            predict_y = model(test_x.float()).detach()
            predict_y = torch.argmax(predict_y, dim=-1)
            current_correct_num = predict_y == test_label
            all_correct_num += np.sum(current_correct_num.to('cpu').numpy(), axis=-1)
            all_sample_num += current_correct_num.shape[0]
        acc = all_correct_num / all_sample_num

        # Writing data on the csv
        data=[current_epoch, acc]
        writer.writerow(data)

        print('Val. accuracy: {:.3f}'.format(acc), flush=True)
        os.makedirs("model/pretrained_models", exist_ok=True)
        if acc > prev_acc:
            torch.save(model.state_dict(), "model/pretrained_models/" + filename + ".pt")
            print(f'Saving')
            counter = 0
            prev_acc = acc
        else:
            counter += 1

print("Model finished training")

test_accuracy(model,
              test_loader,
              device)

