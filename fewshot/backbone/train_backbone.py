import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import network
import numpy as np
import collections
from backbone_utils import get_configuration, get_dataloader, get_tqdm, get_val_dataloader

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # specify which GPU(s) to be used

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=1, type=int, help='seed for training')
parser.add_argument("--dataset", choices=['mini', 'tiered', 'cub'], type=str)
parser.add_argument("--backbone", choices=['resnet18', 'wideres'], type=str, help='network architecture')
parser.add_argument('--epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('--batch_size', default=256, type=int, help='batch size')
parser.add_argument('--silent', action='store_true', help='call --silent to disable tqdm')

args = parser.parse_args()


def main():
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    data_path, split_path, save_path, num_classes = get_configuration(args.dataset, args.backbone)
    train_loader = get_dataloader(data_path, split_path, args.batch_size)
    val_loader = get_val_dataloader(data_path, split_path, 'val')

    model = network.__dict__[args.backbone](num_classes=num_classes)
    model = torch.nn.DataParallel(model).cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[int(.5 * args.epochs), int(.75 * args.epochs)], gamma=0.1)

    tqdm_epochs = get_tqdm(range(args.epochs), args.silent)
    if not args.silent:
        tqdm_epochs.set_description('Total Epochs')

    if not os.path.isdir('../saved_models'):
        os.makedirs('../saved_models')

    best_acc = 0
    for epoch in tqdm_epochs:
        train(train_loader, model, optimizer, epoch)
        scheduler.step()

        val_acc = validate(val_loader, model)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)


def train(train_loader, model, optimizer, epoch):
    model.train()

    correct_count = 0
    total_count = 0
    acc = 0
    tqdm_train_loader = get_tqdm(train_loader, args.silent)

    for batch_idx, (inputs, labels) in enumerate(tqdm_train_loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = outputs.argmax(dim=1)
        correct_count += pred.eq(labels).sum().item()
        total_count += len(inputs)
        acc = correct_count / total_count * 100

        if not args.silent:
            tqdm_train_loader.set_description('Acc {:.2f}'.format(acc))

    if args.silent:
        print("Epoch={}, Accuracy={:.2f}".format(epoch + 1, acc))


# Below codes only used for validation
def validate(val_loader, model):
    embedded_feature = get_embedded_feature(val_loader, model)
    acc_list = []
    tqdm_test_iter = get_tqdm(range(10000), args.silent)
    for _ in tqdm_test_iter:
        acc = nearest_prototype(embedded_feature)
        acc_list.append(acc)

        if not args.silent:
            tqdm_test_iter.set_description('Validate on few-shot tasks. Accuracy:{:.2f}'.format(np.mean(acc_list)))
    if args.silent:
        print("Validate Accuracy={:.2f}".format(np.mean(acc_list)))

    return np.mean(acc_list)


def get_embedded_feature(val_loader, model):
    model.eval()
    with torch.no_grad():
        embedded_feature = collections.defaultdict(list)
        for i, (inputs, labels) in enumerate(val_loader):
            features, _ = model(inputs, return_feature=True)
            features = features.cpu().data.numpy()
            for feature, label in zip(features, labels):
                embedded_feature[label.item()].append(feature)
    return embedded_feature


def nearest_prototype(embedded_feature):
    train_data, test_data, train_label, test_label = sample_task(embedded_feature)

    prototype = train_data.reshape((5, 1, -1)).mean(axis=1)
    distance = np.linalg.norm(prototype - test_data[:, None], axis=-1)

    idx = np.argmin(distance, axis=1)
    pred = np.take(np.unique(train_label), idx)
    acc = (pred == test_label).mean() * 100
    return acc


def sample_task(embedded_feature):
    sample_class = random.sample(list(embedded_feature.keys()), 5)
    train_data, test_data, test_label, train_label = [], [], [], []

    for i, each_class in enumerate(sample_class):
        samples = random.sample(embedded_feature[each_class], 1 + 15)

        train_label += [i] * 1
        test_label += [i] * 15
        train_data += samples[:1]
        test_data += samples[1:]

    return np.array(train_data), np.array(test_data), np.array(train_label), np.array(test_label)


if __name__ == '__main__':
    main()
