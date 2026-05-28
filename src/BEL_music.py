import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from brian2 import *
import warnings

# Set warning filter to ignore specific warning types
warnings.filterwarnings("ignore", message="Bad key", category=UserWarning)

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

# Load data
data = pd.read_excel('expanded-music-data.xlsx', engine='openpyxl')

# Extract features and target variable
X = data[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']]
y = data['EPP']

# Feature normalization
X_normalized = (X - X.min()) / (X.max() - X.min())

# Set time series depth
depth = 2
n = len(X_normalized) - depth
data_input = np.zeros((n, depth, X_normalized.shape[1]))  # Modify data_input to include features
target = np.zeros((n, 1))
output = np.zeros((n, 1))

# Construct time series data
for i in range(n):
    for j in range(depth):
        data_input[i, j] = X_normalized.iloc[i + j].values
    target[i] = y.iloc[i + depth]

# Extract auditory cortex simulation results
M_PYR, M_PV, M_SOM = setup_and_run(data_input)

# Use neuron spike counts as input features
auditory_features = np.array([len(M_PYR.t), len(M_PV.t), len(M_SOM.t)], dtype=float).reshape(1, -1)

# Reshape auditory_features to match RNN input shape
rnn_input = np.tile(auditory_features, (n, 1))

# Simulate thalamus and sensory cortex
thalamus_output = np.random.uniform(0, 1, size=(n, 2))  # Simulate thalamus output signals
cortex_input = thalamus_output  # Sensory cortex receives and processes thalamus output

# Simulate amygdala
amygdala_input = np.concatenate((cortex_input, thalamus_output), axis=1)  # Amygdala receives input from sensory cortex and thalamus
excitatory_system_output = np.zeros((n, 1))
inhibitory_system_output = np.zeros((n, 1))

# Simulate excitatory learning system
for i in range(n):
    excitatory_system_output[i] = np.sum(amygdala_input[i])  # Simple summation as simulation

# Simulate inhibitory output system
for i in range(n):
    inhibitory_system_output[i] = np.sum(amygdala_input[i])  # Simple summation as simulation

# Simulate orbitofrontal cortex
orbitofrontal_input = np.concatenate((cortex_input, amygdala_input), axis=1)  # Orbitofrontal cortex receives input from other cortical regions
orbitofrontal_output = np.zeros((n, 1))

# Simulate inhibitory function of orbitofrontal cortex
for i in range(n):
    orbitofrontal_output[i] = np.sum(orbitofrontal_input[i])  # Simple summation as simulation

# RNN parameters
hidden_size = 10
eta = 0.000000000000001
eta_m = 0.00000000000004
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
        # Get input and target data
        input_data = rnn_input[i]
        target_data = target[i]

        x, y = input_data[-1], input_data[-2]  # Get two feature values from the last time step
        z = target_data[0]  # Target value

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

    # Calculate model accuracy on test set for current epoch
    if iter == epoch - 1:  # Only display accuracy in the last epoch
        correct = 0
        for i in range(number_test):
            x, y = rnn_input[number_train + i, -2], rnn_input[number_train + i, -1]

            z = target[number_train + i][0]  # Target value

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
        output_normalized = (output - np.min(output)) / (np.max(output) - np.min(output))

        # Plot comparison between predicted and original values
        plt.plot(range(number_test), output_normalized[number_train:], label='Predicted')
        plt.plot(range(number_test), target[number_train:], label='Original EPP')
        plt.title('Predicted vs. Original EPP Values')
        plt.xlabel('Data Index')
        plt.ylabel('EPP')
        plt.legend()
        plt.grid(True)
        plt.show()

        # Plot weight heatmaps
        plt.figure(figsize=(12, 4))

        # vi weight heatmap
        plt.subplot(1, 3, 1)
        plt.title('vi weight')
        plt.imshow(vi, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.clim(-1, 1)  # Set colorbar range for consistency across matrices
        plt.xlabel('column')
        plt.ylabel('row')

        # wi weight heatmap
        plt.subplot(1, 3, 2)
        plt.title('wi weight')
        plt.imshow(wi, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.clim(-1, 1)
        plt.xlabel('column')
        plt.ylabel('row')

        # we weight heatmap
        plt.subplot(1, 3, 3)
        plt.title('we weight')
        plt.imshow(we, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.clim(-1, 1)
        plt.xlabel('column')
        plt.ylabel('row')

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

        with open('trained_model-music_parameters.pkl', 'wb') as file:
            pickle.dump(model_parameters, file)