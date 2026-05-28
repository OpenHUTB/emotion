import pandas as pd
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from brian2 import *
import matplotlib.colors as mcolors

# Define the MLP model class
class MultiModalMLP(nn.Module):
    def __init__(self, input_size_audio, input_size_visual, hidden_size=64):
        super(MultiModalMLP, self).__init__()
        self.fc_audio = nn.Linear(input_size_audio, hidden_size)
        self.fc_visual = nn.Linear(input_size_visual, hidden_size)
        self.fc_final = nn.Linear(hidden_size * 2, 1)  # Fusion output layer

    def forward(self, x_audio, x_visual):
        out_audio = torch.relu(self.fc_audio(x_audio))
        out_visual = torch.relu(self.fc_visual(x_visual))
        combined = torch.cat((out_audio, out_visual), dim=1)  # Concatenate audio and visual features
        final_output = self.fc_final(combined)
        return final_output, combined  # Return fusion output and fused feature vector


# Load and preprocess audio features
data_audio = pd.read_excel('music-test111.xlsx', engine='openpyxl')
X_audio = data_audio[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']].values
y_audio = data_audio['EPP'].values.reshape(-1, 1)

scaler_audio = MinMaxScaler()
X_audio_scaled = scaler_audio.fit_transform(X_audio)

# Load and preprocess visual features
data_visual = pd.read_excel('animation-test111.xlsx', engine='openpyxl')
X_visual = data_visual[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values
y_visual = data_visual['EPP'].values.reshape(-1, 1)

scaler_visual = MinMaxScaler()
X_visual_scaled = scaler_visual.fit_transform(X_visual)

# Concatenate audio and visual features
X_combined = np.concatenate((X_audio_scaled, X_visual_scaled), axis=1)
y_combined = np.mean(np.concatenate((y_audio, y_visual), axis=1), axis=1).reshape(-1, 1)

# Combine data into a PyTorch Dataset
class MultiModalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Initialize MLP model and define training parameters
input_size_audio = X_audio.shape[1]
input_size_visual = X_visual.shape[1]
model = MultiModalMLP(input_size_audio, input_size_visual, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 32

# Prepare data loader
dataset = MultiModalDataset(X_combined, y_combined)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        inputs, targets = batch
        audio_features = inputs[:, :input_size_audio]
        visual_features = inputs[:, input_size_audio:]
        optimizer.zero_grad()
        outputs, combined = model(audio_features, visual_features)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Save the trained model
torch.save(model.state_dict(), 'multimodal_mlp_model.pth')

class Flatten(nn.Module):
    """
    Helper module for flattening input tensor to 1-D for the use in Linear modules
    """
    def forward(self, x):
        return x.view(x.size(0), -1)

class Identity(nn.Module):
    """
    Helper module for storing the tensor. Useful for accessing by name
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
        self.output = Identity()

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
data1 = pd.read_excel('animation-test111.xlsx')

# Extract features from the dataset
features = data1[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values

# Convert numpy array to PyTorch tensor and reshape to match model input
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(2).unsqueeze(3)

# Perform inference using CORnet_Z model
model = CORnet_Z()
output = model(features_tensor)

# Apply the output to the input of the subsequent code
X = output.detach().numpy()

# Set random seed for reproducibility
np.random.seed(21)

def setup_and_run(data_input):
    # Define neuron models
    eqs_e = '''
    dv/dt = (I - v) / (10*ms) : volt
    I : volt
    '''
    eqs_p = '''
    dv/dt = (I - v) / (10*ms) : volt
    I : volt
    '''
    eqs_s = '''
    dv/dt = (I - v) / (10*ms) : volt
    I : volt
    '''

    # Create neuron groups
    G_PYR = NeuronGroup(400, eqs_e, threshold='v>0.5*volt', reset='v=0*volt')
    G_PV = NeuronGroup(200, eqs_p, threshold='v>0.5*volt', reset='v=0*volt')
    G_SOM = NeuronGroup(200, eqs_s, threshold='v>0.5*volt', reset='v=0*volt')

    # Set input current for simulation
    G_PYR.I = '0.6 * volt'
    G_PV.I = '0.6 * volt'
    G_SOM.I = '0.65 * volt'

    # Set spike monitors
    M_PYR = SpikeMonitor(G_PYR)
    M_PV = SpikeMonitor(G_PV)
    M_SOM = SpikeMonitor(G_SOM)

    run(1000*ms)  # Run simulation

    return M_PYR, M_PV, M_SOM

# Load data
data2 = pd.read_excel('music-test111.xlsx', engine='openpyxl')

# Extract features and target variable
X = data2[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']]
y = data2['EPP']

# Feature normalization
X_normalized = (X - X.min()) / (X.max() - X.min())

# Set time series depth
depth = 2
n = len(X_normalized) - depth
data_input = np.zeros((n, depth, X_normalized.shape[1]))
target = np.zeros((n, 1))
output = np.zeros((n, 1))

# Construct time series data
for i in range(n):
    for j in range(depth):
        data_input[i, j] = X_normalized.iloc[i + j].values
    target[i] = y.iloc[i + depth]

# Extract auditory cortex simulation results
M_PYR, M_PV, M_SOM = setup_and_run(data_input)

# Use spike counts as input features
auditory_features = np.array([len(M_PYR.t), len(M_PV.t), len(M_SOM.t)], dtype=float).reshape(1, -1)

# Reshape features for RNN input
rnn_input = np.tile(auditory_features, (n, 1))

# Load expanded datasets
data3 = pd.read_excel('expanded-music-data.xlsx', engine='openpyxl')
data4 = pd.read_excel('expanded-animation-data.xlsx', engine='openpyxl')

# Extract input features and target outputs
X1 = data3[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']]
X2 = data4[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']]
y1 = data3['EPP']
y2 = data4['EPP']

# Data normalization
X1 = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
X2 = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2))

# Convert to numpy arrays
X1 = X1.values
X2 = X2.values
y1 = y1.values.reshape(-1, 1)
y2 = y2.values.reshape(-1, 1)

# Fuse two datasets
X = np.concatenate((X1, X2), axis=1)
y = np.mean(np.concatenate((y1, y2), axis=1), axis=1).reshape(-1, 1)

# Set time series depth
depth = 2
n = len(X_combined) - depth
data_input = np.zeros((n, depth, X_combined.shape[1]))
target = np.zeros((n, 1))

# Construct time series data
for i in range(n):
    for j in range(depth):
        data_input[i, j] = X_combined[i + j]
    target[i] = y_combined[i + depth]

data = data_input

# Simulate thalamus and sensory cortex
thalamus_output = np.random.uniform(0, 1, size=(n, 2))
cortex_input = thalamus_output

# Simulate amygdala
amygdala_input = np.concatenate((cortex_input, thalamus_output), axis=1)
excitatory_system_output = np.zeros((n, 1))
inhibitory_system_output = np.zeros((n, 1))

# Simulate excitatory learning system
for i in range(n):
    excitatory_system_output[i] = np.sum(amygdala_input[i])

# Simulate inhibitory output system
for i in range(n):
    inhibitory_system_output[i] = np.sum(amygdala_input[i])

# Simulate orbitofrontal cortex
orbitofrontal_input = np.concatenate((cortex_input, amygdala_input), axis=1)
orbitofrontal_output = np.zeros((n, 1))

# Simulate inhibitory function of orbitofrontal cortex
for i in range(n):
    orbitofrontal_output[i] = np.sum(orbitofrontal_input[i])

# RNN parameters
hidden_size = 10
eta = 0.0004
eta_m = 0.00045
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

# Training process
for iter in range(epoch):
    for i in range(number_train):
        x, y = data[i][-1][-1], data[i][-1][-2]
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

        # Update orbitofrontal cortex weights
        we += eta_m * (Ai[i].reshape(-1, 1) - theta) * Oi[i].reshape(1, -1)

    # Calculate test accuracy in the last epoch
    if iter == epoch - 1:
        correct = 0
        for i in range(number_test):
            x, y = data[number_train + i][-1][-2], data[number_train + i][-1][-1]
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

        # Normalize predicted values
        target_min = np.min(target)
        target_max = np.max(target)
        output_normalized = (output - target_min) / (target_max - target_min)

        # Plot weight heatmaps
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))

        # vi weight heatmap
        ax = axes[0, 0]
        im = ax.imshow(vi, cmap='rainbow', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('vi')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')

        # wi weight heatmap
        ax = axes[0, 1]
        im = ax.imshow(wi, cmap='rainbow', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('wi')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')

        # we weight heatmap
        ax = axes[0, 2]
        im = ax.imshow(we, cmap='rainbow', aspect='auto', vmin=-1, vmax=1)
        ax.set_title('we')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')

        # Ai parameter heatmap
        ax = axes[1, 0]
        im = ax.imshow(Ai, cmap='rainbow', aspect='auto')
        ax.set_title('Ai')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')

        # Oi parameter heatmap
        ax = axes[1, 1]
        im = ax.imshow(Oi, cmap='rainbow', aspect='auto')
        ax.set_title('Oi')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Column index')
        ax.set_ylabel('Row index')

        # Normalized output heatmap
        ax = axes[1, 2]
        im = ax.imshow(output_normalized, cmap='rainbow', aspect='auto')
        ax.set_title('out_n')
        fig.colorbar(im, ax=ax)
        ax.set_xlabel('Feature index')
        ax.set_ylabel('Sample index')
        
        plt.tight_layout()
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

        with open('trained_model-ma_parameters.pkl', 'wb') as file:
            pickle.dump(model_parameters, file)