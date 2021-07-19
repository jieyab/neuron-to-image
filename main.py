import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import time
from neuralnetwork import NeuronActivityDataset, NeuralNet, ConvNet


# define a MSE error computing for testing validation
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# define a r2 computing for testing validation
def r2_score(output, target):
    target_mean = torch.mean(target)
    ss_tot = torch.sum((target - target_mean) ** 2)
    ss_res = torch.sum((target - output) ** 2)
    r2 = 1 - ss_res / ss_tot
    return r2


def train(n_fold, k_fold):
    # setting parameters
    num_epochs = 60
    n_neurons = 500
    n_pixels = 1024
    batch_size = 67
    hidden_size = 1024

    # starting learning_rate
    learning_rate = 0.005
    # a list of loss to trace
    loss_during_train = np.empty((300, 1))
    lr_during_train = np.empty((300, 1))
    # count for saving data
    count_training = 0

    # get training data
    dataset = NeuronActivityDataset(n_fold, k_fold)

    # get testing data
    test_x_data, test_y_data = dataset.get_testing_data()

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)

    # define model
    # model = NeuralNet(n_neurons, hidden_size, n_pixels)
    model = ConvNet(n_neurons, n_pixels)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    # adaptive learning rate using schedule
    scheduler1 = lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.95)
    total_training_samples = dataset.n_samples - dataset.k_fold_samples
    n_iterations = math.ceil(total_training_samples / batch_size)
    print('current fold', n_fold)
    # print the mode info for the first time
    if n_fold == 1:
        print('model info', model)
        print('train model size',len(dataset))

    # start training
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # training process
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            # loss for fcnn
            # loss = criterion(outputs, labels)
            # loss for cnn
            loss = criterion(outputs, labels.view(67, 4, 16, 16))

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # using adaptive lr
            scheduler1.step()

            # print and save some data every 500
            if (i + 1) % 500 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_iterations}], Loss: {loss.item():.4f}')
                loss_during_train[count_training] = loss.item()
                count_training += 1
                print(f'learning rate:', optimizer.state_dict()['param_groups'][0]['lr'])

    np.save("MSE_during_training_fold_cnn_" + str(n_fold), loss_during_train)
    return (model, test_x_data, test_y_data)


def validation(model, test_x_data, test_y_data):
    # testing data
    test_size = len(test_y_data)
    print('test size', test_size)
    # output list
    predicted_pixels = np.empty((test_size, 1024))
    test_loss_list = []  # mse loss list
    r2_list = []  # r2 score list
    count = 0
    with torch.no_grad():
        for test_x, test_y in list(zip(test_x_data, test_y_data)):
            outputs = model(test_x)
            outputslist = outputs.data.tolist()
            predicted_pixels[count] = outputslist
            count += 1
            loss = mse(outputs, test_y)
            test_loss_list.append(loss)

            # r2 score
            r2 = r2_score(outputs, test_y)
            r2_list.append(r2)

    print('average predicted loss', np.mean(test_loss_list), "r2 length", len(r2_list), np.mean(r2_list))


if __name__ == '__main__':
    xy = pd.read_csv('PhDData_500n_1rep40000n_img.csv')
    network = 'cnn'  # network to try cnn, fcnn, unet
    k_fold = 5
    for n_fold in range(1, k_fold + 1):
        model, test_x_data, test_y_data = train(n_fold, k_fold)
        validation(model, test_x_data, test_y_data)
