# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:46:22 2020

@author: ssridhar
"""

def fetch_data_step():
    import torch
    import torchvision
    import torchvision.transforms as transforms
    
    """The output of torchvision datasets are PILImage images of range [0, 1].
    We transform them to Tensors of normalized range [-1, 1].
    """
    
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                              shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    return trainset, trainloader, testset, testloader, classes