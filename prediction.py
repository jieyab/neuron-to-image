import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# load training data
xy = pd.read_csv('PhDData_500n_1rep40000n_img.csv')
# load  data for prediction
prediction_data = pd.read_csv('PhD_neuron_output.csv')

class NeuronActivityDataset(Dataset):
    def __init__(self):
        global xy
        self.n_samples = len(xy)
        self.neurons_activity = xy.iloc[:, 1024:]
        self.pixel_values = xy.iloc[:, : 1024]
        self.x_data = torch.Tensor(self.neurons_activity.values)  # size [n_training_samples, n_features]
        self.y_data = torch.Tensor(self.pixel_values.values)  # size [n_training_samples, n_output]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(in_features=input_size, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=1024)
        self.fc3 = nn.Linear(in_features=1024, out_features=output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.sigmoid(self.fc2(out))
        out = self.fc3(out)
        return out

def train():
    # setting parameters
    num_epochs = 40
    n_neurons = 500
    n_pixels = 1024
    batch_size = 16
    hidden_size = 1024

    # static learning_rate
    learning_rate = 0.001  # adapted learning rate start setting is 0.01
    # get training data
    dataset = NeuronActivityDataset()

    # get testing data

    train_loader = DataLoader(dataset=dataset,
                              batch_size=batch_size,
                              shuffle=True)

    # define model
    model = NeuralNet(n_neurons, hidden_size, n_pixels)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    # adaptive learning rate using schedule
    scheduler1 = lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.95)

    # start training
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(train_loader):
            # training process
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            # using adaptive lr
            #scheduler1.step()
    return model


def prediction(model, prediction_size,prediction_dataset):
    # creaste empty output list
    predicted_pixels_list = np.empty((prediction_size, 1024))
    count = 0
    predict_input_tensor = torch.tensor(prediction_dataset.values)
    #load data
    with torch.no_grad():
        for predict_x in predict_input_tensor:
            predict_outputs = model(predict_x.float())
            predicted_pixels = predict_outputs.data.tolist()
            predicted_pixels_list[count] = predicted_pixels
            count += 1
    return predicted_pixels_list

# train the model using whole training data
model = train()
#save model
FILE = 'fcnnmodel.pth'
model.eval()
prediction_size = prediction_data.shape[0]
predicted_pixels_list = prediction(model,prediction_size,prediction_data)
np.save("predicted pixels_", predicted_pixels_list)

#plot one image
image1 = np.reshape(predicted_pixels_list[0],(32,32))
plt.imshow(image1, 'gray', origin='lower')
plt.show()