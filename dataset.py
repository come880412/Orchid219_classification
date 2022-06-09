from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

import argparse
import os

import numpy as np
from PIL import Image
import cv2

transforms_test =   transforms.Compose([
                    transforms.Resize((384, 384)),
                    # transforms.Resize((672, 672)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.423, 0.367), (0.247, 0.241, 0.249)),])

transforms_test_640 =   transforms.Compose([
                    transforms.Resize((640, 640)),
                    # transforms.Resize((672, 672)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.423, 0.367), (0.247, 0.241, 0.249)),])

transforms_train =  transforms.Compose([
                    transforms.Resize((384, 384)),
                    # transforms.Resize((672, 672)),
                    # transforms.RandomResizedCrop(size=(384, 384), scale=(0.2, 1.)),
                    # transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(0.4),
                    transforms.RandomVerticalFlip(0.4),
                    transforms.RandomApply(torch.nn.ModuleList([
                        transforms.ColorJitter(brightness=0.4, saturation=0.4, contrast=0.4, hue=0.1),
                    ]), p=0.4),
                    transforms.RandomApply(torch.nn.ModuleList([
                        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), scale=(0.5, 1.5)),
                    ]), p=0.4),
                    transforms.ToTensor(),
                    transforms.Normalize((0.480, 0.423, 0.367), (0.247, 0.241, 0.249)),
                    transforms.RandomErasing(0.25)
                    ])

class Orchid_data(Dataset):
    def __init__(self, opt, mode):
        self.root = opt.root
        self.mode = mode
        self.model = opt.model
        self.split_type = opt.split_type
        self.k = opt.k

        if mode == 'train':
            self.transform = transforms_train
        elif mode == 'valid' or mode == 'test':
            self.transform = transforms_test                                                          
        
        if self.split_type == "random":
            self.data_info = np.loadtxt(os.path.join(self.root, "random_split", self.mode + ".csv"), delimiter=',', dtype=np.str)[1:]
        
        elif self.split_type == "k_fold":
            self.data_info = np.loadtxt(os.path.join(self.root, "k_fold", self.k, self.mode + ".csv"), delimiter=',', dtype=np.str)[1:]

    def __getitem__(self,index):
        image_name, label = self.data_info[index]
        image_path = os.path.join(self.root, "images", image_name)

        image = Image.open(image_path).convert("RGB")
        return self.transform(image), transforms_test_640(image), int(label), image_name

    def __len__(self):
        return len(self.data_info)

class Orchid_pretrain_data(Dataset):
    def __init__(self, opt, mode):
        self.root = opt.root

        self.data = []

        if mode == 'train':
            self.transform = transforms_train
        elif mode == 'valid' or mode == 'test':
            self.transform = transforms_test
        
        data_info = np.loadtxt(os.path.join(self.root, "Species_Classifier", mode + '.txt'), delimiter=',', dtype=np.str)

        for info in data_info:
            image_name, label = info

            image_path = os.path.join(self.root, 'Orchid_Images', image_name)
            label = int(label) - 1

            self.data.append([image_path, label])

    def __getitem__(self, index):
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image), label, image_path

    def __len__(self):
        return len(self.data)

class Orchid_public(Dataset):
    def __init__(self, opt):
        self.root = opt.root

        self.data = []
        sample = np.loadtxt(os.path.join(self.root, 'submission_template.csv'), delimiter=',', dtype=np.str)[1:]
        
        for image_name, _ in sample:
            image_path = os.path.join(self.root, 'private_and_public', image_name)
            self.data.append([image_path, image_name])

    def __getitem__(self, index):
        image_path, image_name = self.data[index]
        image = Image.open(image_path).convert('RGB')

        return transforms_test(image), transforms_test_640(image), image_name
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
    parser.add_argument("--initial_epoch", type=int, default=0, help="Start epoch")
    parser.add_argument("--lr_decay_epoch", type=int, default=30, help="Start to decay epoch")

    parser.add_argument('--root', default='../dataset', help='path to dataset')
    parser.add_argument('--split_type', default='random', help='random, k_fold')
    parser.add_argument('--k', default='1', help='Which fold you want to use')
    
    parser.add_argument('--num_classes', type=int, default=219, help='number of classes')
    parser.add_argument('--optimizer', default='sgd', help='adam/sgd')
    parser.add_argument('--scheduler', default='step', help='linear/step')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=2, help="batch_size")

    parser.add_argument('--model', default='resnet18', help='resnet18/resnet50')

    parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu workers')
    parser.add_argument('--load', default='', help='path to model to continue training')
    parser.add_argument('--save_model', default='./checkpoints', help='path to save model')
    
    opt = parser.parse_args()
    dataset = Orchid_pretrain_data(opt, 'train')
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=2,shuffle=True, num_workers=0)
    iter_data = iter(train_loader)
    image, label = iter_data.next()
    
    print(image.shape, label)