import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math
import time
from torch.optim.optimizer import required
from torch.utils.data.dataset import random_split
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from pytorchtools import EarlyStopping




def train_test_ktree(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60):
    '''
    Trains and tests k-tree models
    Inputs: model, trainloader, validloader, testloader, epochs, randorder, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    # Initialize loss function and optimizer
#     criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    # if randorder == True, generate the randomizer index array for randomizing the input image pixel order
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(epochs):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()

            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
#             if torch.max(outputs) == 1:
#                 print('max output 1')
#                 print(torch.unique(outputs))
                
            loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            if torch.sum(torch.isnan(loss)) > 0:
                break
            loss.backward()
            
####        # Freeze select weights by zeroing out gradients
            for child in model.children():
                for param in child.parameters():
                    for freeze_mask in model.freeze_mask_set:
                        if hasattr(param.grad, 'shape'):
                            if param.grad.shape == freeze_mask.shape:
                                param.grad[freeze_mask] = 0
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += ((outputs > 0) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            # Generate loss and accuracy curves by saving average every 4th minibatch
            if (i % 4) == 3:    
                loss_curve.append(running_loss/4)
                acc_curve.append(running_acc/4)
                running_loss = 0.0
                running_acc = 0.0
        
        if torch.sum(torch.isnan(loss)) > 0:
            print('loss is nan, now testing')
            break
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for _, data in enumerate(validloader):
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = criterion(output + 1e-8, labels.float().reshape(-1,1))
            # record validation loss
            valid_losses.append(loss.item())
                
        valid_loss = np.average(valid_losses)


        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # load the last checkpoint with the best model
#    model.load_state_dict(torch.load('checkpoint.pt'))
    
    print('Finished Training, %d epochs' % (epoch+1))
    
    ######################    
    # test the model     #
    ######################    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                images = images[:,ordering].cuda()
            else:
                images = images.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            # Sum up correct labelings
            predicted = outputs > 0
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()
    # Calculate test accuracy
    accuracy = correct/total
    
    print('Accuracy of the network on the test images: %2f %%' % (
        100 * accuracy))
    print('final outputs:', torch.unique(outputs))
    if randorder == True:
        return(loss_curve, acc_curve, loss, accuracy, model, ordering)
    else:
        return(loss_curve, acc_curve, loss, accuracy, model)

def train_test_fc(model, trainloader, validloader, testloader, epochs=10, patience=60, lr=0.001):
    '''
    Trains and tests fcnn models
    Inputs: model, trainloader, validloader, testloader, epochs, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    
    t = Timer()
    t.start()
    
    # Initialize loss function and optimizer
#     criterion = nn.BCELoss()
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)
        
        
    for epoch in range(epochs):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs + 1e-10, labels.float().reshape(-1,1))
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
#             running_acc += (torch.round(outputs) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            running_acc += ((outputs > 0) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size

            if i % 4 == 3:      # Generate loss and accuracy curves by saving average every 4th minibatch
                loss_curve.append(running_loss/4)
                acc_curve.append(running_acc/4)
                running_loss = 0.0
                running_acc = 0.0
            
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for _, data in enumerate(validloader):
            inputs, labels, _ = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = criterion(output + 1e-8, labels.float().reshape(-1,1))
            # record validation loss
            valid_losses.append(loss.item())
                
        valid_loss = np.average(valid_losses)


        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    # load the last checkpoint with the best model
#    model.load_state_dict(torch.load('checkpoint.pt'))
    
    print('Finished Training, %d epochs' % (epoch+1))
    
    correct = 0
    all_loss = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            images = images.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            # Sum up correct labelings
            predicted = outputs > 0
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()
            all_loss += loss
    # Calculate test accuracy
    accuracy = correct/total
    # Calculate average loss
    ave_loss = all_loss.item()/total
    if ave_loss > 1000000:
            print('ave_loss = ', ave_loss)
            ave_loss = 4
            print('ave_loss = ', ave_loss)
    print('Accuracy of the network on the 10000 test images: %4f %%' % (
        100 * accuracy))
    t.stop()
    return(loss_curve, acc_curve, ave_loss, accuracy, model)


class TimerError(Exception):
    """A custom exception used to report errors in use of Timer class"""

class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")

        self._start_time = time.perf_counter()

    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        
        
def train_test_ktree_sparse(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60,
                            lr=0.001):
    '''
    Trains and tests k-tree models
    Inputs: model, trainloader, validloader, testloader, epochs, randorder, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    t = Timer()
    t.start()
    
    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    # if randorder == True, generate the randomizer index array for randomizing the input image pixel order
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(epochs):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()

            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
                
            loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            if torch.sum(torch.isnan(loss)) > 0:
                break
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += ((outputs > 0) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            # Generate loss and accuracy curves by saving average every 4th minibatch
            if (i % 4) == 3:    
                loss_curve.append(running_loss/4)
                acc_curve.append(running_acc/4)
                running_loss = 0.0
                running_acc = 0.0
        
        if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(outputs)) > 0:
            print('loss is nan, now testing')
            break
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for _, data in enumerate(validloader):
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs)
            # calculate the loss
            loss = criterion(output + 1e-8, labels.float().reshape(-1,1))
            # record validation loss
            valid_losses.append(loss.item())
                
        valid_loss = np.average(valid_losses)


        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        if epoch > 200:
            early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
            
#         if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(outputs)) > 0:
#             loss = 10
#             accuracy = 0.5
#             return(loss_curve, acc_curve, loss, accuracy, model)
    # load the last checkpoint with the best model
#    model.load_state_dict(torch.load('checkpoint.pt'))
    
    print('Finished Training, %d epochs' % (epoch+1))
    
    ######################    
    # test the model     #
    ######################    
    correct = 0
    total = 0
    all_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                images = images[:,ordering].cuda()
            else:
                images = images.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            # Sum up correct labelings
            predicted = outputs > 0
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()
            all_loss += loss
    # Calculate test accuracy
    accuracy = correct/total
    # Calculate average loss
    ave_loss = all_loss.item()/total
    
    print('Accuracy of the network on the test images: %2f %%' % (
        100 * accuracy))
#     print('final outputs:', torch.unique(outputs))
    
    t.stop()
#     if torch.sum(torch.isnan(torch.Tensor(ave_loss))) > 0 or torch.sum(torch.isnan(outputs)) > 0:
#             ave_loss = 10
# #             accuracy = 0.5
#             return(loss_curve, acc_curve, ave_loss, accuracy, model)

    if np.sum(np.isnan(np.array(ave_loss))) > 0:
        print('nan ave_loss = ', ave_loss)
        ave_loss = 4
        print('ave_loss = ', ave_loss)
    if ave_loss > 4:
        print('big ave_loss = ', ave_loss)
        ave_loss = 4
        print('ave_loss = ', ave_loss)
    if randorder == True:
        return(loss_curve, acc_curve, ave_loss, accuracy, model, ordering)
    else:
        return(loss_curve, acc_curve, ave_loss, accuracy, model)
    
def train_test_ktree_sparse_debug(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60,
                            lr=0.001):
    loss_curve = []
    acc_curve = []
    ave_loss = 4
    accuracy = 1
    model = []
    return(loss_curve, acc_curve, ave_loss, accuracy, model)
    
def train_test_ktree_multistage(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60,
                                lr=0.001, multistage=True, stages=[0,1,2]):
    '''
    Trains and tests k-tree models
    Inputs: model, trainloader, validloader, testloader, epochs, randorder, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    t = Timer()
    t.start()
    
    syn_layers = []
    for syn_name in model.syn_names:
        for syn_layer in list(model._modules[syn_name].parameters()):
            syn_layers.append(syn_layer)
    den_layers = []
    for repeat in range(model.Repeats):
        for den_name in model.names[repeat]:
            for den_layer in list(model._modules[den_name].parameters()):
                den_layers.append(den_layer)
    sqgl_nonlin = list(model.sqgl.parameters())
    
    # Initialize loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # if randorder == True, generate the randomizer index array for randomizing the input image pixel order
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    if multistage == False:
        stages = [3]

    for stage in stages:
        
        # Initialize loss function and optimizer
        if stage == 0:
            optimizer = optim.Adam(syn_layers, lr=lr)
        elif stage == 1:
            optimizer = optim.Adam(den_layers, lr=lr)
        elif stage == 2:
            optimizer = optim.Adam(sqgl_nonlin, lr=lr)
        else:
            optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # to track training loss and accuracy as model trains
        loss_curve = []
        acc_curve = []

        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []



        # Initialize early stopping object
        early_stopping = EarlyStopping(patience=patience, verbose=False)

        for epoch in range(epochs):  # loop over the dataset multiple times
            ######################    
            # train the model    #
            ######################
            running_loss = 0.0
            running_acc = 0.0
            model.train()

            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels, _ = data
                if randorder == True:
                    # Randomize pixel order
                    inputs = inputs[:,ordering].cuda()
                else:
                    inputs = inputs.cuda()

                labels = labels.cuda()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)

                loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
                if torch.sum(torch.isnan(loss)) > 0:
                    break
                loss.backward()

                optimizer.step()

                # print statistics
                running_loss += loss.item()
                running_acc += ((outputs > 0) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
                # Generate loss and accuracy curves by saving average every 4th minibatch
                if (i % 4) == 3:    
                    loss_curve.append(running_loss/4)
                    acc_curve.append(running_acc/4)
                    running_loss = 0.0
                    running_acc = 0.0

            if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(outputs)) > 0:
                print('loss is nan, now testing')
                break
            ######################    
            # validate the model #
            ######################
            model.eval() # prep model for evaluation
            for _, data in enumerate(validloader):
                inputs, labels, _ = data
                if randorder == True:
                    # Randomize pixel order
                    inputs = inputs[:,ordering].cuda()
                else:
                    inputs = inputs.cuda()
                labels = labels.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(inputs)
                # calculate the loss
                loss = criterion(output + 1e-8, labels.float().reshape(-1,1))
                # record validation loss
                valid_losses.append(loss.item())

            valid_loss = np.average(valid_losses)


            # early_stopping needs the validation loss to check if it has decreased, 
            # and if it has, it will make a checkpoint of the current model
            if epoch > 200:
                early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(outputs)) > 0:
                loss = 10
                accuracy = 0.5
                return(loss_curve, acc_curve, loss, accuracy, model)
        # load the last checkpoint with the best model
    #    model.load_state_dict(torch.load('checkpoint.pt'))

        print('Finished Training, %d epochs' % (epoch+1))
    
    ######################    
    # test the model     #
    ######################    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                images = images[:,ordering].cuda()
            else:
                images = images.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            # Sum up correct labelings
            predicted = outputs > 0
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()
    # Calculate test accuracy
    accuracy = correct/total
    
    print('Accuracy of the network on the test images: %2f %%' % (
        100 * accuracy))
#     print('final outputs:', torch.unique(outputs))
    
    t.stop()
    if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(outputs)) > 0:
            loss = 10
#             accuracy = 0.5
            return(loss_curve, acc_curve, loss, accuracy, model)
    if randorder == True:
        return(loss_curve, acc_curve, loss, accuracy, model, ordering)
    else:
        return(loss_curve, acc_curve, loss, accuracy, model)

def train_test_ktree_synapse(model, trainloader, validloader, testloader, epochs=10, randorder=False, patience=60,
                            lr=0.001):
    '''
    Trains and tests k-tree models
    Inputs: model, trainloader, validloader, testloader, epochs, randorder, patience
    Outputs: train loss_curve, train acc_curve, test ave_loss, test accuracy, trained model
    '''
    t = Timer()
    t.start()
    
    # Initialize loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # to track training loss and accuracy as model trains
    loss_curve = []
    acc_curve = []
    
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    
    # if randorder == True, generate the randomizer index array for randomizing the input image pixel order
    if randorder == True:
        ordering = torch.randperm(len(trainloader.dataset.tensors[0][0]))
    
    # Initialize early stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    for epoch in range(epochs):  # loop over the dataset multiple times
        ######################    
        # train the model    #
        ######################
        running_loss = 0.0
        running_acc = 0.0
        model.train()
        
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()

            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs, loss_model = model(inputs)
            
            loss_pred = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            loss = loss_pred + loss_model
            if torch.sum(torch.isnan(loss)) > 0:
                break
            loss.backward()
            
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            running_acc += ((outputs > 0) == labels.float().reshape(-1,1)).sum().item()/trainloader.batch_size
            # Generate loss and accuracy curves by saving average every 4th minibatch
            if (i % 4) == 3:    
                loss_curve.append(running_loss/4)
                acc_curve.append(running_acc/4)
                running_loss = 0.0
                running_acc = 0.0
        
        if torch.sum(torch.isnan(loss)) > 0 or torch.sum(torch.isnan(outputs)) > 0:
            print('loss is nan, now testing')
            break
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for _, data in enumerate(validloader):
            inputs, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                inputs = inputs[:,ordering].cuda()
            else:
                inputs = inputs.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output, loss_model = model(inputs)
            # calculate the loss
            loss_pred = criterion(output + 1e-8, labels.float().reshape(-1,1))
            loss = loss_pred + loss_model
            # record validation loss
            valid_losses.append(loss.item())
                
        valid_loss = np.average(valid_losses)


        # early_stopping needs the validation loss to check if it has decreased, 
        # and if it has, it will make a checkpoint of the current model
        if epoch > 200:
            early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break
            
    
    print('Finished Training, %d epochs' % (epoch+1))
    
    ######################    
    # test the model     #
    ######################    
    correct = 0
    total = 0
    all_loss = 0
    with torch.no_grad():
        for data in testloader:
            images, labels, _ = data
            if randorder == True:
                # Randomize pixel order
                images = images[:,ordering].cuda()
            else:
                images = images.cuda()
            labels = labels.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs, loss_model = model(images)
            # calculate the loss
            loss_pred = criterion(outputs + 1e-8, labels.float().reshape(-1,1))
            
            loss = loss_pred + loss_model
            # Sum up correct labelings
            predicted = outputs > 0
            total += labels.size(0)
            correct += (predicted == labels.float().reshape(-1,1)).sum().item()
            all_loss += loss
    # Calculate test accuracy
    accuracy = correct/total
    # Calculate average loss
    ave_loss = all_loss.item()/total
    
    print('Accuracy of the network on the test images: %2f %%' % (
        100 * accuracy))
#     print('final outputs:', torch.unique(outputs))
    
    t.stop()

    if np.sum(np.isnan(np.array(ave_loss))) > 0:
        print('nan ave_loss = ', ave_loss)
        ave_loss = 4
        print('ave_loss = ', ave_loss)
    if ave_loss > 4:
        print('big ave_loss = ', ave_loss)
        ave_loss = 4
        print('ave_loss = ', ave_loss)
    if randorder == True:
        return(loss_curve, acc_curve, ave_loss, accuracy, model, ordering)
    else:
        return(loss_curve, acc_curve, ave_loss, accuracy, model)