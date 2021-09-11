#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split


from conf import Settings
from utils import network
from dataset import PanNukEx, Records



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()
    settings = Settings()

    dataset = PanNukEx(
        settings.lmdb_path,
    )

    print(len(dataset))
    data_size = len(dataset)
    train_size = int(data_size * 0.9)
    valid_size = data_size - train_size
    train_dataset, valid_dataset = random_split(dataset, [train_size, valid_size],
            generator=torch.Generator().manual_seed(42))

    mean = dataset.mean
    std = dataset.std
    valid_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((settings.image_size, settings.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    validation_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.b, num_workers=4, shuffle=False)
    validation_loader.dataset.dataset.transforms = valid_transforms


    net = network(args.net, dataset.num_classes, pretrained=False)

    net.load_state_dict(torch.load(args.weights))
    net.cuda()
    print(net)
    net.eval()

    correct_1 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(validation_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(validation_loader)))

            image = image.cuda()
            print(image.shape)
            label = label.cuda()

            output = net(image)
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(validation_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
