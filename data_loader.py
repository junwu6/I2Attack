from torchvision import datasets, transforms
import torch
import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset


def make_dataset(image_list, labels):
    if labels:
      len_ = len(image_list)
      images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
      if len(image_list[0].split()) > 2:
        images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
      else:
        images = [('E:/Codes'+val.split()[0], int(val.split()[1])) for val in image_list]   # update root
    return images


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


class ImageList(Dataset):
    def __init__(self, image_list, labels=None, transform=None, target_transform=None, mode='RGB'):
        imgs = make_dataset(image_list, labels)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        if mode == 'RGB':
            self.loader = rgb_loader
        elif mode == 'L':
            self.loader = l_loader

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

class DatasetSplit(torch.utils.data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def load_training(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.RandomCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor()])
    data = ImageList(open(root_path + dir).readlines(), transform=transform)
    train_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False, drop_last=False, num_workers=4)
    return train_loader


def load_testing(root_path, dir, batch_size):
    transform = transforms.Compose(
        [transforms.Resize([224, 224]),
         transforms.ToTensor()])
    data = ImageList(open(root_path + dir).readlines(), transform=transform)
    test_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False, num_workers=4)
    return test_loader


def load_training_svhn2mnist(root_path, dir):
    transform = None
    if 'svhn' in dir:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif 'mnist' in dir:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif 'usps' in dir:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    data = ImageList(open(root_path + dir).readlines(), transform=transform, mode='RGB')
    train_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False, drop_last=False, num_workers=4)
    return train_loader


def load_testing_svhn2mnist(root_path, dir):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    data = ImageList(open(root_path + dir).readlines(), transform=transform, mode='RGB')
    test_loader = torch.utils.data.DataLoader(data, batch_size=len(data), shuffle=False, num_workers=4)
    return test_loader

