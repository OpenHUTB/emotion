import pandas as pd
import pickle
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from brian2 import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
# Define the MLP model class
# Cross-modal fusion layer definition
class CrossModalFusionLayer(nn.Module):
    def __init__(self, input_size_audio, input_size_visual, hidden_size=64):
        super(CrossModalFusionLayer, self).__init__()
        self.fc_audio = nn.Linear(input_size_audio, hidden_size)
        self.fc_visual = nn.Linear(input_size_visual, hidden_size)
        self.attn_audio = nn.Linear(hidden_size, hidden_size)
        self.attn_visual = nn.Linear(hidden_size, hidden_size)
        self.fc_final = nn.Linear(hidden_size * 2, 1)  # Final output layer

    def forward(self, x_audio, x_visual):
        # Project audio and visual features to hidden space
        out_audio = torch.relu(self.fc_audio(x_audio))
        out_visual = torch.relu(self.fc_visual(x_visual))

        # Calculate attention weights for audio and visual features
        audio_attention = torch.softmax(self.attn_audio(out_audio), dim=-1)
        visual_attention = torch.softmax(self.attn_visual(out_visual), dim=-1)

        # Apply attention to features
        out_audio = out_audio * audio_attention
        out_visual = out_visual * visual_attention

        # Combine features
        combined = torch.cat((out_audio, out_visual), dim=1)
        final_output = self.fc_final(combined)
        return final_output, combined


# Define the MultiModalMLP model class
class MultiModalMLP(nn.Module):
    def __init__(self, input_size_audio, input_size_visual, hidden_size=64):
        super(MultiModalMLP, self).__init__()
        self.fusion_layer = CrossModalFusionLayer(input_size_audio, input_size_visual, hidden_size)

    def forward(self, x_audio, x_visual):
        return self.fusion_layer(x_audio, x_visual)


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

# Combine audio and visual features
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


# Initialize MLP model with Cross-modal Fusion layer
input_size_audio = X_audio.shape[1]
input_size_visual = X_visual.shape[1]
model = MultiModalMLP(input_size_audio, input_size_visual, hidden_size=64)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 100
batch_size = 32

# Prepare DataLoader
dataset = MultiModalDataset(X_combined, y_combined)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
for epoch in range(epochs):
    for batch in dataloader:
        inputs, targets = batch
        audio_features = inputs[:, :input_size_audio]
        visual_features = inputs[:, input_size_audio:]

        optimizer.zero_grad()
        outputs, combined = model(audio_features, visual_features)  # Use Cross-modal Fusion Layer
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Print loss for every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Save the trained model
torch.save(model.state_dict(), 'multimodal_mlp_model_with_fusion.pth')


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

# Example: Loading and preparing the audio and visual data
data1 = pd.read_excel('animation-test111.xlsx')

features = data1[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values
features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(2).unsqueeze(3)
# Perform inference using CORnet_Z model
model = CORnet_Z()
output = model(features_tensor)

# 将输出结果应用于第二段代码的输入
X = output.detach().numpy()

# 设置随机数生成器的种子
np.random.seed(21)
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

# 读取数据
data2 = pd.read_excel('music-test111.xlsx', engine='openpyxl')

# 提取特征和目标变量
X = data2[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']]
y = data2['EPP']

# 特征归一化处理
X_normalized = (X - X.min()) / (X.max() - X.min())

# 设置时间序列深度
depth = 2
n = len(X_normalized) - depth
data_input = np.zeros((n, depth, X_normalized.shape[1]))  # 修改 data_input 以包含特征
target = np.zeros((n, 1))
output = np.zeros((n, 1))

# 构建时间序列数据
for i in range(n):
    for j in range(depth):
        data_input[i, j] = X_normalized.iloc[i + j].values
    target[i] = y.iloc[i + depth]

# 提取听觉皮层模拟结果
M_PYR, M_PV, M_SOM = setup_and_run(data_input)

# 调试打印 SpikeMonitor 中的事件时间长度

# 使用神经元脉冲数量作为输入特征
auditory_features = np.array([len(M_PYR.t), len(M_PV.t), len(M_SOM.t)], dtype=float).reshape(1, -1)

# 将 auditory_features 调整为循环神经网络的输入形状
rnn_input = np.tile(auditory_features, (n, 1))

# 数据加载
data3 = pd.read_excel('music-test111.xlsx', engine='openpyxl')
data4 = pd.read_excel('animation-test111.xlsx', engine='openpyxl')

# 提取输入特征和预测输出
X1 = data3[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']]
X2 = data4[['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']]
y1 = data3['EPP']
y2 = data4['EPP']

# 归一化处理
X1 = (X1 - np.min(X1)) / (np.max(X1) - np.min(X1))
X2 = (X2 - np.min(X2)) / (np.max(X2) - np.min(X2))

# 将数据转换为numpy数组
X1 = X1.values
X2 = X2.values
y1 = y1.values.reshape(-1, 1)
y2 = y2.values.reshape(-1, 1)

# 将两份数据进行融合
X = np.concatenate((X1, X2), axis=1)
y = np.mean(np.concatenate((y1, y2), axis=1), axis=1).reshape(-1, 1)

# 数据准备部分
depth = 2
n = len(X_combined) - depth
data_input = np.zeros((n, depth, X_combined.shape[1]))  # 修改 data_input 以包含特征
target = np.zeros((n, 1))

# 构建时间序列数据
for i in range(n):
    for j in range(depth):
        data_input[i, j] = X_combined[i + j]
    target[i] = y_combined[i + depth]
data_input = torch.tensor(data_input, dtype=torch.float32)
target = torch.tensor(target, dtype=torch.float32)


# Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size

        # Encoder 部分
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2, dropout=dropout),
            num_layers=num_layers
        )

        # Decoder 部分 (一个简单的线性层)
        self.decoder = nn.Linear(hidden_size, 1)

        # 输入到 Transformer 的嵌入层
        self.embedding = nn.Linear(input_size, hidden_size)

    def forward(self, x):
        # 输入的形状为 (batch_size, sequence_length, input_size)
        x = self.embedding(x)  # Shape: (n, depth, hidden_size)

        # 转置为 (sequence_length, batch_size, hidden_size)
        x = x.permute(1, 0, 2)

        # Transformer Encoder 处理
        x = self.encoder(x)

        # 解码（预测值）
        x = x[-1, :, :]  # 获取最后一个时间步的输出
        x = self.decoder(x)

        return x


# 超参数
hidden_size = 10
eta = 0.00004
eta_m = 0.000045
eta_o = 0.05  # 眶额皮层学习率
theta = 0.5  # 眶额皮层阈值
rew = 2
epoch = 100
number_train = round(0.75 * n)
number_test = n - number_train
# 初始化权重
vi = np.random.uniform(-1, 1, size=(1, 2))
wi = np.random.uniform(-1, 1, size=(1, 2))
Ai = np.zeros((n, 2))
Oi = np.zeros((n, 2))
we = np.random.uniform(-1, 1, size=(2, 2))

# 创建 Transformer 模型
model = TransformerModel(input_size=X_combined.shape[1], hidden_size=hidden_size)
optimizer = optim.Adam(model.parameters(), lr=eta)
criterion = nn.MSELoss()

# 记录训练和测试过程中的准确率
accuracies = []

# 训练过程
for iter in range(epoch):
    model.train()
    for i in range(number_train):
        # 获取数据
        x = data_input[i]
        y = target[i]

        # 前向传播
        optimizer.zero_grad()
        output = model(x.unsqueeze(0))  # 添加 batch 维度
        loss = criterion(output, y)

        # 反向传播
        loss.backward()
        optimizer.step()

    # 测试准确率
    if iter == epoch - 1:
        correct = 0
        model.eval()
        with torch.no_grad():
            for i in range(number_test):
                x = data_input[number_train + i]
                y = target[number_train + i]
                output = model(x.unsqueeze(0))  # 添加 batch 维度

                if np.sign(output.item()) == np.sign(y.item()):
                    correct += 1

        accuracy = correct / number_test * 100
        print(f'Epoch {iter + 1}, 测试准确率: {accuracy:.2f}%')

        # 提取目标变量的最小值和最大值
        target_min = torch.min(target).item()  # 获取最小值
        target_max = torch.max(target).item()  # 获取最大值

        # 归一化预测值
        output_normalized = (output - target_min) / (target_max - target_min)

        # 使用pickle保存模型参数
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

        with open('trained_model-mat_parameters.pkl', 'wb') as file:
            pickle.dump(model_parameters, file)

