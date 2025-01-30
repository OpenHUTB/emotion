import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
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

# 对归一化后的真实 EPP 和生成 EPP 进行排序
sorted_real_EPP = sorted(real_EPP_values)
sorted_generated_EPP_normalized = sorted(data['Generated_EPP_Normalized'])

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
