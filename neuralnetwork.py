import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
import time

xy = pd.read_csv('PhDData_500n_1rep40000n_img.csv')


class NeuronActivityDataset(Dataset):
    def __init__(self, test_fold=1, k_fold=6):
        global xy
        self.n_samples = len(xy)
        # testing sample size for k fold
        self.k_fold_samples = math.floor(self.n_samples / k_fold)
        # training sample size for k fold
        self.n__training_samples = self.n_samples - self.k_fold_samples

        # spliting data to get training set
        self.test_split_beginning = (test_fold - 1) * self.k_fold_samples  # fold are label from 0 to k_fold -1
        self.test_split_end = test_fold * self.k_fold_samples
        # neurons_activity = xy.iloc[np.r_[:self.test_split_beginning -1, self.test_split_end: ], 1024:] # getting the training data
        # pixel_values = xy.iloc[np.r_[:self.test_split_beginning, self.test_split_end:], : 1024]
        self.neurons_activitybegin = xy.iloc[:self.test_split_beginning, 1024:]
        self.neurons_activityend = xy.iloc[self.test_split_end:, 1024:]
        self.neurons_activity = pd.concat([self.neurons_activitybegin, self.neurons_activityend], axis=0,
                                          ignore_index=True)

        self.pixel_valuesbegin = xy.iloc[:self.test_split_beginning, : 1024]
        self.pixel_valuesend = xy.iloc[self.test_split_end:, : 1024]
        self.pixel_values = pd.concat([self.pixel_valuesbegin, self.pixel_valuesend], axis=0, ignore_index=True)

        self.x_data = torch.Tensor(self.neurons_activity.values)  # size [n_training_samples, n_features]
        self.y_data = torch.Tensor(self.pixel_values.values)  # size [n_training_samples, n_output]

    def get_testing_data(self):
        global xy
        test_x_data = torch.Tensor(xy.iloc[self.test_split_beginning:self.test_split_end, 1024:].values)
        test_y_data = torch.Tensor(xy.iloc[self.test_split_beginning:self.test_split_end, :1024].values)
        return test_x_data, test_y_data

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n__training_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.sigmoid(self.fc2(out))
        out = torch.sigmoid(self.fc3(out))
        return out


class ConvNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(ConvNet, self).__init__()
        self.fc1 = nn.Linear(in_features=input_size, out_features=18 * 18)
        # input tensor should be shape (batch size =1, input feature size=.., 1) or  (1,1,1024)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3)


    def forward(self, x):
        print(x.size())
        out = torch.sigmoid(self.fc1(x))
        print(out.size())
        out = out.view(67, 1, 18, 18)
        out = torch.sigmoid(self.conv1(out))
        #print('cov1 works',out.size(),"flatten", torch.flatten(out, start_dim=1).shape)

        # out = self.conv2(out)
        return out
