import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import euclidean_distances

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
# 加载模型参数
with open('trained_model-animation_parameters.pkl', 'rb') as file:
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
data = pd.read_excel('animation-test111.xlsx', engine='openpyxl')
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
    features = row[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values
    real_EPP = row['EPP']

    generated_EPP = generate_EPP(features)

    generated_EPP_values.append(generated_EPP)
    real_EPP_values.append(real_EPP)

# 计算最小和最大值
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

# 计算欧氏距离并百分比化
euclidean_distances_array = [(1 - euclidean_distances(np.reshape(generated_EPP, (1, -1)), np.reshape(real_EPP, (1, -1)))[0][0] /
                              (max_generated_EPP - min_generated_EPP)) * 100 for generated_EPP, real_EPP in
                             zip(generated_EPP_values, real_EPP_values)]

# 计算平均欧氏距离百分比
average_euclidean_distance_percentage = np.mean(euclidean_distances_array)

# 将生成的EPP值添加到数据框中
data['Generated_EPP'] = generated_EPP_values

# 归一化生成的EPP值
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

data['Generated_EPP_Normalized'] = (generated_EPP_values - min_generated_EPP) / (max_generated_EPP - min_generated_EPP)
# 打印生成值和真实值的对比数值及欧氏距离
for i in range(len(data)):
    if euclidean_distances_array[i] >= 0:
        print(f"Real EPP: {real_EPP_values[i]:.6f}, Generated EPP (Normalized): {data['Generated_EPP_Normalized'].iloc[i]:.6f}, Euclidean Distance: {euclidean_distances_array[i]:.2f}")

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
plt.title('A-BEL model Smoothed Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()


# Load model parameters from file (this part needs to be integrated as per your specific model parameters)
# with open('trained_model-animation_parameters.pkl', 'rb') as file:
#     model_parameters = pickle.load(file)
# vi = model_parameters['vi']
# wi = model_parameters['wi']
# we = model_parameters['we']
# eta_o = model_parameters['eta_o']
# theta = model_parameters['theta']
# depth = model_parameters['depth']
# eta = model_parameters['eta']
# eta_m = model_parameters['eta_m']
# rew = model_parameters['rew']

# Generate EPP values and calculate metrics
def generate_EPP(features):
    x, y = features[0], features[1]
    max_s = max(x, y)

    # Assuming some calculation here for Ai, Oi, E, and applying cortical inhibition
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
    euclidean_distance = \
    euclidean_distances(np.reshape(generated_EPP_values[i], (1, -1)), np.reshape(real_EPP_values[i], (1, -1)))[0][0]
    print(
        f"Real EPP: {real_EPP_values[i]:.6f}, Generated EPP (Normalized): {generated_EPP_values_normalized[i]:.6f}, Euclidean Distance: {euclidean_distance:.2f}")
