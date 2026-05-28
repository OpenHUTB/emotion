import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import torch
from torch import nn

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    """
    Helper module that stores the current tensor. Useful for accessing by name
    """
    def forward(self, x):
        return x

class CORblock_Z(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        self.nonlin = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.output = Identity()  # for an easy access to this block's output

    def forward(self, inp):
        x = self.conv(inp)
        x = self.nonlin(x)
        x = self.pool(x)
        x = self.output(x)
        return x

def CORnet_Z():
    model = nn.Sequential(
        CORblock_Z(5, 64, kernel_size=7, stride=2),
        CORblock_Z(64, 128),
        CORblock_Z(128, 256),
        CORblock_Z(256, 512),
        nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(512, 1000),
            Identity()
        )
    )

    # Weight initialization
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return model

# Load data from Excel file
data = pd.read_excel('animation-test111.xlsx')

# Assuming data is in the form of features (bpm, jitter, consonance, bigsmall, updown)
# You may need to adjust the data processing steps based on the actual structure of your data
features = data[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values

# Convert NumPy array to PyTorch tensor and adjust the shape to match the model's input shape
# Assuming the number of samples is 760
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(2).unsqueeze(3)

# Perform inference using CORnet_Z model
model = CORnet_Z()
output = model(features_tensor)

# Apply the output results to the input of the second code segment
X = output.detach().numpy()

# Second code segment
# Set the seed for the random number generator
np.random.seed(21)

# Data loading
data = pd.read_excel('animation-test111.xlsx', engine='openpyxl')

# Extract input features and target output
y = data['EPP'].values.reshape(-1, 1)

# Set time series depth
depth = 2
n = len(X) - depth
data_input = np.zeros((n, depth, X.shape[1]))  # Modify data_input to include features
target = np.zeros((n, 1))
output = np.zeros((n, 1))

# Construct time series data
for i in range(len(X) - depth):
    for j in range(depth):
        data_input[i, j] = X[i + j]  # Store feature sequences
    target[i] = y[i + depth]

data = data_input

# Simulate thalamus and sensory cortex components
thalamus_output = np.random.uniform(0, 1, size=(n, 2))  # Simulate thalamus output signals
cortex_input = thalamus_output  # Sensory cortex receives thalamus output and processes it

# Simulate amygdala component
amygdala_input = np.concatenate((cortex_input, thalamus_output), axis=1)  # Amygdala receives input from sensory cortex and thalamus
excitatory_system_output = np.zeros((n, 1))
inhibitory_system_output = np.zeros((n, 1))

# Simulate excitatory learning system
for i in range(n):
    excitatory_system_output[i] = np.sum(amygdala_input[i])  # Simple summation as simulation

# Simulate inhibitory output system
for i in range(n):
    inhibitory_system_output[i] = np.sum(amygdala_input[i])  # Simple summation as simulation

# Simulate orbitofrontal cortex component
orbitofrontal_input = np.concatenate((cortex_input, amygdala_input), axis=1)  # Orbitofrontal cortex receives input from other cortical regions
orbitofrontal_output = np.zeros((n, 1))

# Simulate inhibitory function of the orbitofrontal cortex
for i in range(n):
    orbitofrontal_output[i] = np.sum(orbitofrontal_input[i])  # Simple summation as simulation

# Recurrent neural network parameters
hidden_size = 10
eta = 0.00000001
eta_m = 0.00000005
eta_o = 0.05  # Orbitofrontal cortex learning rate
theta = 0.5  # Orbitofrontal cortex threshold
rew = 2

# Initialize weights
vi = np.random.uniform(-1, 1, size=(1, 2))
wi = np.random.uniform(-1, 1, size=(1, 2))
Ai = np.zeros((n, 2))
Oi = np.zeros((n, 2))
we = np.random.uniform(-1, 1, size=(2, 2))

# Training parameters
epoch = 100
number_train = round(0.75 * n)
number_test = n - number_train

# Record accuracy for each epoch
accuracies = []

# Training loop
for iter in range(epoch):
    for i in range(number_train):
        x, y = data[i][-1][-1], data[i][-1][-2]  # Modified line
        z = target[i]

        max_s = max(x, y)
        Ai[i, 0] = x * vi[0, 0]
        Ai[i, 1] = y * vi[0, 1]
        Oi[i, 0] = x * wi[0, 0]
        Oi[i, 1] = y * wi[0, 1]
        E = (np.sum(Ai[i]) + max_s) - np.sum(Oi[i])
        error = z - E

        delta_vi = error * eta * np.array([[x * max(0, rew - np.sum(Ai[i]))], [y * max(0, rew - np.sum(Ai[i]))]])
        vi += delta_vi.T

        wi[0, 0] += error * eta_m * (x * (Oi[i, 0] + Oi[i, 1] - 2 * rew))
        wi[0, 1] += error * eta_m * (y * (Oi[i, 0] + Oi[i, 1] - 2 * rew))

        # Update orbitofrontal cortex
        we += eta_m * (Ai[i].reshape(-1, 1) - theta) * Oi[i].reshape(1, -1)

    # Calculate model accuracy on the test set for current epoch
    if iter == epoch - 1:  # Only display accuracy in the last epoch
        correct = 0
        for i in range(number_test):
            x, y = data[number_train + i][-1][-2], data[number_train + i][-1][-1]  # Modified line
            max_s = max(x, y)
            Ai[i, 0] = x * vi[0, 0]
            Ai[i, 1] = y * vi[0, 1]
            Oi[i, 0] = x * wi[0, 0]
            Oi[i, 1] = y * wi[0, 1]
            E = (np.sum(Ai[i]) + max_s) - np.sum(Oi[i])
            output[number_train + i] = E - np.sum(we * Ai[i])

            if np.sign(output[number_train + i]) == np.sign(target[number_train + i]):
                correct += 1

        accuracy = correct / number_test * 100
        print(f'Epoch {iter + 1}, Test Accuracy: {accuracy:.2f}%')

        # Extract min and max values of the target variable
        target_min = np.min(target)
        target_max = np.max(target)

        # Normalize predicted values
        output_normalized = (output - target_min) / (target_max - target_min)

        # Normalize predicted values
        output_normalized = (output - np.min(output)) / (np.max(output) - np.min(output))

        # Plotting the comparison between predicted values and original EPP values
        plt.plot(range(number_test), output_normalized[number_train:], label='Predicted')
        plt.plot(range(number_test), target[number_train:], label='Original EPP')
        plt.title('Comparison between Predicted and Original EPP Values')
        plt.xlabel('Data Index')
        plt.ylabel('EPP')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Save model parameters using pickle
        model_parameters = {
            'vi': vi,
            'wi': wi,
            'we': we,
            'eta_o': eta_o,
            'theta': theta,
            'depth': depth,
            'eta': eta,
            'eta_m': eta_m,
            'rew': rew
        }

        with open('trained_model-animationt_parameters.pkl', 'wb') as file:
            pickle.dump(model_parameters, file)