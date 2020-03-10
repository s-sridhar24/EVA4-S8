# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 11:22:08 2020

@author: ssridhar
"""

def train_net(epoch, device, net, trainloader, optimizer, criterion, train_batch_size):
    running_loss = 0.0
    mini_batch = 2000 * 4/train_batch_size
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % mini_batch == (mini_batch - 1):    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / mini_batch))
            running_loss = 0.0
