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
def setup_and_run(data_input):
    # 定义神经元模型
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

    # 创建神经元组
    G_PYR = NeuronGroup(400, eqs_e, threshold='v>0.5*volt', reset='v=0*volt')
    G_PV = NeuronGroup(200, eqs_p, threshold='v>0.5*volt', reset='v=0*volt')
    G_SOM = NeuronGroup(200, eqs_s, threshold='v>0.5*volt', reset='v=0*volt')

    # 运行仿真
    G_PYR.I = '0.6 * volt'  # 设置输入电流
    G_PV.I = '0.6 * volt'
    G_SOM.I = '0.65 * volt'

    # 设置监视器
    M_PYR = SpikeMonitor(G_PYR)
    M_PV = SpikeMonitor(G_PV)
    M_SOM = SpikeMonitor(G_SOM)

    run(1000*ms)  # 运行仿真一段时间

    return M_PYR, M_PV, M_SOM

# 加载模型参数
with open('trained_model-music_parameters.pkl', 'rb') as file:
    model_parameters = pickle.load(file)

# 提取模型参数
vi = model_parameters['vi']
wi = model_parameters['wi']
we = model_parameters['we']
eta_o = model_parameters['eta_o']
theta = model_parameters['theta']
depth = model_parameters['depth']
eta = model_parameters['eta']
eta_m = model_parameters['eta_m']
rew = model_parameters['rew']

# 从文件加载数据
data = pd.read_excel('music-test111.xlsx', engine='openpyxl')

# 生成程序
def generate_EPP(features):
    x, y = features[0], features[1]
    max_s = max(x, y)

    # 计算Ai和Oi
    Ai = np.zeros((depth,))
    Oi = np.zeros((depth,))
    for j in range(depth):
        Ai[j] = x * vi[0, j]
        Oi[j] = y * wi[0, j]

    # 计算E
    E = (np.sum(Ai) + max_s) - np.sum(Oi)

    # 应用眶额皮层的抑制影响
    E -= np.sum(we * Ai)

    return E

# 遍历所有数据行，生成新的EPP值并归一化
generated_EPP_values = []
real_EPP_values = []

# 初始化最小和最大值
min_generated_EPP = np.inf
max_generated_EPP = -np.inf

# 计算欧氏距离
euclidean_distances_array = []

for index, row in data.iterrows():
    features = row[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']].values
    real_EPP = row['EPP']

    generated_EPP = generate_EPP(features)

    generated_EPP_values.append(generated_EPP)
    real_EPP_values.append(real_EPP)

# 计算最小和最大值
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

# 计算欧氏距离并百分比化
euclidean_distances_array = [(1 - euclidean_distances(np.reshape(generated_EPP, (1, -1)), np.reshape(real_EPP, (1, -1)))[0][0] /
                              (max_generated_EPP - min_generated_EPP)) for generated_EPP, real_EPP in
                             zip(generated_EPP_values, real_EPP_values)]

# 计算平均欧氏距离百分比
average_euclidean_distance_percentage = np.mean(euclidean_distances_array) * 100

# 将生成的EPP值添加到数据框中
data['Generated_EPP'] = generated_EPP_values

# 归一化生成的EPP值
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

data['Generated_EPP_Normalized'] = (generated_EPP_values - min_generated_EPP) / (max_generated_EPP - min_generated_EPP)
# 打印生成值和真实值的对比数值及欧氏距禈
for i in range(len(data)):
    print(
        f"Real EPP: {real_EPP_values[i]:.6f}, Generated EPP (Normalized): {data['Generated_EPP_Normalized'].iloc[i]:.6f}, Euclidean Distance: {euclidean_distances_array[i]:.2f}")

# 输出平均欧氏距离
print(f"\nAverage Euclidean Distance Percentage: {average_euclidean_distance_percentage:.2f}")
# 计算相似度
#similarity = (1 - average_euclidean_distance_percentage / 100)*100

# 输出相似度
#print(f"\nAverage Similarity: {similarity:.2f}%")
# 计算指数函数相似度
exponential_similarity = np.exp(-average_euclidean_distance_percentage / 100) * 100

# 输出指数函数相似度
print(f"\nExponential Similarity: {exponential_similarity:.2f}%")
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
epochs = 200
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
# Plot smoothed training loss curve
plt.figure(figsize=(10, 5))

# Smooth the losses using a moving average with window size 10
smoothed_losses = np.convolve(losses, np.ones(10)/10, mode='valid')

plt.plot(range(1, len(smoothed_losses) + 1), smoothed_losses, label='Smoothed Training Loss')
plt.title('M-BEL model Smoothed Training Loss over Epochs')
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

# Additional plot for neuronal simulation or other visualizations
M_PYR, M_PV, M_SOM = setup_and_run(data_visual)  # Assuming setup_and_run returns SpikeMonitors

plt.figure(figsize=(10, 5))
plt.plot(M_PYR.t/ms, M_PYR.i, 'r.', label='PYR Neurons')
plt.plot(M_PV.t/ms, M_PV.i + 400, 'b.', label='PV Neurons')
plt.plot(M_SOM.t/ms, M_SOM.i + 600, 'g.', label='SOM Neurons')
plt.title('Neuronal Spike Raster Plot')
plt.xlabel('Time (ms)')
plt.ylabel('Neuron Index')
plt.legend()
plt.grid(True)
plt.show()
