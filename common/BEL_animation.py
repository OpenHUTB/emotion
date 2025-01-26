import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_kernels
from brian2 import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

# Define the MLP model class
class MultiModalMLP(nn.Module):
    def __init__(self, input_size_visual, hidden_size=64):
        super(MultiModalMLP, self).__init__()
        self.fc_visual = nn.Linear(input_size_visual, hidden_size)
        self.fc_final = nn.Linear(hidden_size, 1)

    def forward(self, x_visual):
        out_visual = torch.relu(self.fc_visual(x_visual))
        final_output = self.fc_final(out_visual)
        return final_output

# Load and preprocess visual features
data_visual = pd.read_excel('animation-test111.xlsx', engine='openpyxl')
X_visual = data_visual[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values
y_visual = data_visual['EPP'].values.reshape(-1, 1)

scaler_visual = MinMaxScaler()
X_visual_scaled = scaler_visual.fit_transform(X_visual)

# PyTorch Dataset
class MultiModalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Instantiate the model and define training parameters
input_size_visual = X_visual.shape[1]
model = MultiModalMLP(input_size_visual=input_size_visual, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 32

# Prepare data loader
dataset = MultiModalDataset(X_visual_scaled, y_visual)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop with loss collection
losses = []
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch in dataloader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * inputs.size(0)
    epoch_loss /= len(dataset)
    losses.append(epoch_loss)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

# Save trained model
torch.save(model.state_dict(), 'multimodal_mlp_model.pth')

# Plot training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(1, epochs + 1), losses, label='Training Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Load model parameters from file (adjust based on your specific parameters)
with open('trained_model-animation_parameters.pkl', 'rb') as file:
    model_parameters = pickle.load(file)

# Assuming you have setup and_run function for neuronal simulation

# Generate EPP values and calculate metrics (adjust generate_EPP function based on your needs)
def generate_EPP(features):
    x, y = features[0], features[1]
    max_s = max(x, y)

    Ai = np.zeros((depth,))
    Oi = np.zeros((depth,))
    for j in range(depth):
        Ai[j] = x * vi[0, j]
        Oi[j] = y * wi[0, j]

    E = (np.sum(Ai) + max_s) - np.sum(Oi)
    E -= np.sum(we * Ai)

    return E

# Iterate over data rows to generate new EPP values and normalize
generated_EPP_values = []
real_EPP_values = []

for index, row in data_visual.iterrows():
    features = row[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values
    real_EPP = row['EPP']

    generated_EPP = generate_EPP(features)

    generated_EPP_values.append(generated_EPP)
    real_EPP_values.append(real_EPP)

# Normalize generated EPP values
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)
generated_EPP_values_normalized = (generated_EPP_values - min_generated_EPP) / (max_generated_EPP - min_generated_EPP)

# Print comparison and Euclidean distances
for i in range(len(data_visual)):
    euclidean_distance = euclidean_distances(np.reshape(generated_EPP_values[i], (1, -1)), np.reshape(real_EPP_values[i], (1, -1)))[0][0]
    print(f"Real EPP: {real_EPP_values[i]:.6f}, Generated EPP (Normalized): {generated_EPP_values_normalized[i]:.6f}, Euclidean Distance: {euclidean_distance:.2f}")

# Plot comparison of real and generated EPP values
plt.figure(figsize=(10, 5))
plt.plot(data_visual.index, real_EPP_values, marker='o', linestyle='-', color='b', label='Real EPP')
plt.plot(data_visual.index, generated_EPP_values_normalized, marker='o', linestyle='-', color='r', label='Generated EPP (Normalized)')
plt.title('Real EPP vs. Generated EPP (Normalized) with Euclidean Distance')
plt.xlabel('Sample Index')
plt.ylabel('EPP Value')
plt.legend()
plt.grid(True)
plt.show()

# Additional plot for neuronal simulation or other visualizations
M_PYR, M_PV, M_SOM = setup_and_run(data_visual)  # Assuming setup_and_run returns SpikeMonitors

plt.figure(figsize=(10, 5))
plt.plot(M_PYR.t/ms, M_PYR.i, 'r.', label='Excitatory Neurons')
plt.plot(M_PV.t/ms, M_PV.i + 400, 'b.', label='PV Neurons')
plt.plot(M_SOM.t/ms, M_SOM.i + 600, 'g.', label='SOM Neurons')
plt.title('Neuronal Spike Raster Plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.legend()
plt.grid(True)
plt.show()
