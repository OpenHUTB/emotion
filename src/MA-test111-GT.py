import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from itertools import zip_longest
from sklearn.metrics.pairwise import euclidean_distances
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import precision_recall_fscore_support
# Define the Transformer model class
class MultiModalTransformer(nn.Module):
    def __init__(self, input_size_audio, input_size_visual, hidden_size=64, num_heads=4, num_layers=2):
        super(MultiModalTransformer, self).__init__()

        # Embedding layers to project input features into a higher dimension
        self.audio_embedding = nn.Linear(input_size_audio, hidden_size)
        self.visual_embedding = nn.Linear(input_size_visual, hidden_size)

        # Transformer Encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads),
            num_layers=num_layers
        )

        # Final output layer
        self.fc_final = nn.Linear(hidden_size, 1)  # Output EPP value

    def forward(self, x_audio, x_visual):
        # Embedding the input features into hidden space
        audio_embedded = self.audio_embedding(x_audio)
        visual_embedded = self.visual_embedding(x_visual)

        # Combine the audio and visual embeddings into a single input for transformer
        combined_input = torch.cat((audio_embedded, visual_embedded), dim=0)  # Concatenate along the sequence dimension

        # Apply transformer encoder
        transformer_output = self.transformer_encoder(combined_input.unsqueeze(1))  # Add batch dimension

        # Get the output from the last position of the sequence
        final_output = self.fc_final(transformer_output[-1])
        return final_output, transformer_output


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


# Create a Dataset class for PyTorch
class MultiModalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Instantiate the model and define training parameters
input_size_audio = X_audio.shape[1]
input_size_visual = X_visual.shape[1]
model = MultiModalTransformer(input_size_audio, input_size_visual, hidden_size=64, num_heads=4, num_layers=2)
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

        outputs, combined = model(audio_features, visual_features)  # Forward pass through the Transformer
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    # Optionally print loss during training
    # if (epoch + 1) % 10 == 0:
    #     print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# Save trained model
torch.save(model.state_dict(), 'multimodal_transformer_model.pth')

# After training, you can proceed with the rest of the code for generating EPP values
# The code for the rest of the pipeline remains the same as you defined in your original script.
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
with open('trained_model-mat_parameters.pkl', 'rb') as file:
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
data_music = pd.read_excel('music-test111.xlsx', engine='openpyxl')
data_animation = pd.read_excel('animation-test111.xlsx', engine='openpyxl')

# 计算加权平均EPP参考值
# weighted_avg_reference_EPP = (0.3 * data_music['EPP'].mean() + 0.7 * data_animation['EPP'].mean())

# 生成程序
def generate_EPP(features_music, features_animation):
    x_music, y_music = features_music[0], features_music[1]
    x_animation, y_animation = features_animation[0], features_animation[1]

    max_s_music = max(x_music, y_music)
    max_s_animation = max(x_animation, y_animation)

    # 计算 Ai 和 Oi
    Ai_music = np.zeros((depth,))
    Oi_music = np.zeros((depth,))
    for j in range(depth):
        Ai_music[j] = x_music * vi[0, j]
        Oi_music[j] = y_music * wi[0, j]

    Ai_animation = np.zeros((depth,))
    Oi_animation = np.zeros((depth,))
    for j in range(depth):
        Ai_animation[j] = x_animation * vi[0, j]
        Oi_animation[j] = y_animation * wi[0, j]

    # 计算 E
    E_music = (np.sum(Ai_music) + max_s_music) - np.sum(Oi_music)
    E_animation = (np.sum(Ai_animation) + max_s_animation) - np.sum(Oi_animation)

    # 应用眶额皮层的抑制影响
    E_music -= np.sum(we * Ai_music)
    E_animation -= np.sum(we * Ai_animation)

    # 综合两份文件的特征值生成新的 EPP 值
    generated_EPP = np.mean([E_music, E_animation])

    return generated_EPP

# 遍历所有数据行，生成新的 EPP 值并归一化
generated_EPP_values = []
weighted_avg_reference_EPP_list = []
# 初始化最小和最大值
min_generated_EPP = np.inf
max_generated_EPP = -np.inf

# 计算欧氏距离
euclidean_distances_array = []

for index, (row_music, row_animation) in enumerate(zip_longest(data_music.iterrows(), data_animation.iterrows(), fillvalue=(None, None))):
    features_music = row_music[1][['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']].values
    features_animation = row_animation[1][['bpm', 'jitter', 'consonance', 'bigsmall', 'updown']].values

    # 跳过缺失的行
    if row_music[1] is None or row_animation[1] is None:
        print(f"Skipping sample {index + 1} due to missing data.")
        continue

    generated_EPP = generate_EPP(features_music, features_animation)

    generated_EPP_values.append(generated_EPP)

    # 计算加权平均EPP参考值
    weighted_avg_reference_EPP = (0.25 * row_music[1]['EPP'] + 0.75 * row_animation[1]['EPP'])
    weighted_avg_reference_EPP_list.append(weighted_avg_reference_EPP)

# 计算最小和最大值
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

# 在计算欧氏距离之前，将数组转换为列表
weighted_avg_reference_EPP_list = [0.25 * row_music[1]['EPP'] + 0.75 * row_animation[1]['EPP'] for row_music, row_animation in zip(data_music.iterrows(), data_animation.iterrows())]



# 修改这行代码
euclidean_distances_array = [(1 - euclidean_distances(np.array(generated_EPP).reshape(1, -1), np.array(real_EPP).reshape(1, -1))[0][0] /
                              (max_generated_EPP - min_generated_EPP)) * 100 for generated_EPP, real_EPP in
                             zip_longest(generated_EPP_values, weighted_avg_reference_EPP_list, fillvalue=(None, None))]

# 计算平均欧氏距离百分比
average_euclidean_distance_percentage = np.mean(euclidean_distances_array)

# 将生成的 EPP 值添加到数据框中
data_music['Generated_EPP'] = generated_EPP_values

# 归一化生成的 EPP 值
min_generated_EPP = np.min(generated_EPP_values)
max_generated_EPP = np.max(generated_EPP_values)

data_music['Generated_EPP_Normalized'] = (generated_EPP_values - min_generated_EPP) / (max_generated_EPP - min_generated_EPP)

# 打印生成值和真实值的对比数值及欧氏距离
for i in range(len(data_music)):
    if euclidean_distances_array[i] >= 0:
      print(
        f"Real EPP: {weighted_avg_reference_EPP_list[i]:.6f}, Generated EPP (Normalized): {data_music['Generated_EPP_Normalized'].iloc[i]:.6f}, Euclidean Distance: {euclidean_distances_array[i]:.2f}")

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
# Convert EPP values to binary labels if needed (for illustration purposes)
# Example: Convert to binary labels based on a threshold
threshold = 0.2  # Adjust threshold as per your requirement

# Example: Assuming binary classification based on threshold
generated_labels = np.where(data_music['Generated_EPP_Normalized'] > threshold, 1, 0)
real_labels = np.where(data_music['EPP'] > threshold, 1, 0)
# Compute precision, recall, and F1-score
precision, recall, f1_score, _ = precision_recall_fscore_support(real_labels, generated_labels, average='binary')
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1_score:.2f}")

# 对归一化后的真实 EPP 和生成 EPP 进行排序
sorted_real_EPP = sorted(weighted_avg_reference_EPP_list)
sorted_generated_EPP_normalized = sorted(data_music['Generated_EPP_Normalized'])

# 创建样条插值模型进行平滑
spline_real = UnivariateSpline(range(len(sorted_real_EPP)), sorted_real_EPP, s=0.5)  # s是平滑因子
spline_generated = UnivariateSpline(range(len(sorted_generated_EPP_normalized)), sorted_generated_EPP_normalized, s=0.5)

# 生成插值后的平滑曲线
x_smooth = np.linspace(0, len(sorted_real_EPP)-1, 500)
y_smooth_real = spline_real(x_smooth)
y_smooth_generated = spline_generated(x_smooth)

# 计算拟合关系（使用最小二乘法拟合真实 EPP 和生成 EPP）
slope, intercept = np.polyfit(x_smooth, y_smooth_generated, 1)  # 线性拟合

# 生成拟合后的趋势线
y_smooth_trendline = slope * x_smooth + intercept

# 绘制平滑的归一化真实 EPP 和生成 EPP 的趋近曲线
plt.plot(x_smooth, y_smooth_real, linestyle='-', color='b', label=' Real EPP')
plt.plot(x_smooth, y_smooth_generated, linestyle='-', color='r', label='Generated EPP')

# 绘制拟合的趋势线（表示两者的线性趋近关系）
plt.plot(x_smooth, y_smooth_trendline, linestyle='--', color='g', label='Fitted Trend Line')

plt.title(' Real EPP vs. Generated EPP(Transformer)')
plt.xlabel('Sorted Sample Index')
plt.ylabel('EPP Value')
plt.legend()
plt.grid(True)
plt.show()