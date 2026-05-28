import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support

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

threshold = 0.25  # Adjust threshold as per your requirement

# Load model parameters
with open('trained_model-animation_parameters.pkl', 'rb') as file:
    model_parameters = pickle.load(file)

# Extract model parameters
vi = model_parameters['vi']
wi = model_parameters['wi']
we = model_parameters['we']
eta_o = model_parameters['eta_o']
theta = model_parameters['theta']
depth = model_parameters['depth']
eta = model_parameters['eta']
eta_m = model_parameters['eta_m']
rew = model_parameters['rew']

# Load data from Excel file
data = pd.read_excel('animation-test111.xlsx', engine='openpyxl')

# EPP generation function
def generate_EPP(features):
    x, y = features[0], features[1]
    max_s = max(x, y)

    # Calculate Ai and Oi values
    Ai = np.zeros((depth,))
    Oi = np.zeros((depth,))
    for j in range(depth):
        Ai[j] = x * vi[0, j]
        Oi[j] = y * wi[0, j]

    # Calculate E value
    E = (np.sum(Ai) + max_s) - np.sum(Oi)

    # Apply inhibitory influence from orbitofrontal cortex
    E -= np.sum(we * Ai)

    return E

# Iterate all data rows, generate new EPP values and prepare for normalization
generated_EPP_values = []
real_EPP_values = []

# Initialize minimum and maximum values
min_generated_EPP = np.inf
max_generated_EPP = -np.inf

# Calculate Euclidean distances
euclidean_distances_array = []

for index, row in data.iterrows():
    features = row[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values
    real_EPP = row['EPP']

    generated_EPP = generate_EPP(features)

    generated_EPP_values.append(generated_EPP)
    real_EPP_values.append(real_EPP)

# Calculate min and max of generated EPP values
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

# Calculate Euclidean distance and convert to percentage
euclidean_distances_array = [(1 - euclidean_distances(np.reshape(generated_EPP, (1, -1)), np.reshape(real_EPP, (1, -1)))[0][0] /
                              (max_generated_EPP - min_generated_EPP)) * 100 for generated_EPP, real_EPP in
                             zip(generated_EPP_values, real_EPP_values)]

# Calculate average Euclidean distance percentage
average_euclidean_distance_percentage = np.mean(euclidean_distances_array)

# Add generated EPP values to dataframe
data['Generated_EPP'] = generated_EPP_values

# Normalize the generated EPP values to [0, 1]
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)
data['Generated_EPP_Normalized'] = (generated_EPP_values - min_generated_EPP) / (max_generated_EPP - min_generated_EPP)

# Print comparison between real and generated EPP values with Euclidean distance
for i in range(len(data)):
    if euclidean_distances_array[i] >= 0:
        print(f"Real EPP: {real_EPP_values[i]:.6f}, Generated EPP (Normalized): {data['Generated_EPP_Normalized'].iloc[i]:.6f}, Euclidean Distance: {euclidean_distances_array[i]:.2f}")

# Print average Euclidean distance
print(f"\nAverage Euclidean Distance Percentage: {average_euclidean_distance_percentage:.2f}")

# Calculate exponential similarity
exponential_similarity = np.exp(-average_euclidean_distance_percentage / 100) * 100

# Print exponential similarity
print(f"\nExponential Similarity: {exponential_similarity:.2f}%")

# Convert EPP values to binary labels if needed (for illustration purposes)
# Example: Convert to binary labels based on a threshold
threshold = 0.2  # Adjust threshold as per your requirement

# Example: Assuming binary classification based on threshold
generated_labels = np.where(data['Generated_EPP_Normalized'] > threshold, 1, 0)
real_labels = np.where(data['EPP'] > threshold, 1, 0)

# Compute precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(real_labels, generated_labels, average='binary')

print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")

# Plot line chart for comparison
plt.plot(data.index, real_EPP_values, marker='o', linestyle='-', color='b', label='Real EPP')
plt.plot(data.index, data['Generated_EPP_Normalized'], marker='o', linestyle='-', color='r',
         label='Generated EPP (Normalized)')
plt.title('Real EPP vs. Generated EPP (Normalized) with Euclidean Distance')
plt.xlabel('Sample Index')
plt.ylabel('EPP Value')
plt.legend()
plt.grid(True)
plt.show()