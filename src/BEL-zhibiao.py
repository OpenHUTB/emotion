import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_kernels
from brian2 import *
from sklearn.metrics import precision_recall_fscore_support

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

    # Run simulation
    G_PYR.I = '0.6 * volt'  # Set input current
    G_PV.I = '0.6 * volt'
    G_SOM.I = '0.65 * volt'

    # Set monitors
    M_PYR = SpikeMonitor(G_PYR)
    M_PV = SpikeMonitor(G_PV)
    M_SOM = SpikeMonitor(G_SOM)

    run(1000*ms)  # Run simulation for a period

    return M_PYR, M_PV, M_SOM

# Load model parameters
with open('trained_model-music_parameters.pkl', 'rb') as file:
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

# Load data from file
data = pd.read_excel('music-test111.xlsx', engine='openpyxl')

# EPP generation function
def generate_EPP(features):
    x, y = features[0], features[1]
    max_s = max(x, y)

    # Calculate Ai and Oi
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
    features = row[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']].values
    real_EPP = row['EPP']

    generated_EPP = generate_EPP(features)

    generated_EPP_values.append(generated_EPP)
    real_EPP_values.append(real_EPP)

# Calculate min and max of generated EPP values
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

# Calculate Euclidean distance and convert to percentage
euclidean_distances_array = [(1 - euclidean_distances(np.reshape(generated_EPP, (1, -1)), np.reshape(real_EPP, (1, -1)))[0][0] /
                              (max_generated_EPP - min_generated_EPP)) for generated_EPP, real_EPP in
                             zip(generated_EPP_values, real_EPP_values)]

# Calculate average Euclidean distance percentage
average_euclidean_distance_percentage = np.mean(euclidean_distances_array) * 100

# Add generated EPP values to dataframe
data['Generated_EPP'] = generated_EPP_values

# Normalize the generated EPP values to [0, 1]
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)
data['Generated_EPP_Normalized'] = (generated_EPP_values - min_generated_EPP) / (max_generated_EPP - min_generated_EPP)

# Print comparison between real and generated EPP values with Euclidean distance
for i in range(len(data)):
    print(
        f"Real EPP: {real_EPP_values[i]:.6f}, Generated EPP (Normalized): {data['Generated_EPP_Normalized'].iloc[i]:.6f}, Euclidean Distance: {euclidean_distances_array[i]:.2f}")

# Print average Euclidean distance
print(f"\nAverage Euclidean Distance Percentage: {average_euclidean_distance_percentage:.2f}")

# Calculate exponential similarity
exponential_similarity = np.exp(-average_euclidean_distance_percentage / 100) * 100

# Print exponential similarity
print(f"\nExponential Similarity: {exponential_similarity:.2f}%")

# Convert EPP values to binary labels for classification evaluation
threshold = 0.202 # Adjust threshold as per your requirement

# Generate binary labels based on threshold
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