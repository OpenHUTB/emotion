import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_kernels
from brian2 import *
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

# 绘制折线图对比
plt.plot(data.index, real_EPP_values, marker='o', linestyle='-', color='b', label='Real EPP')
plt.plot(data.index, data['Generated_EPP_Normalized'], marker='o', linestyle='-', color='r',
         label='Generated EPP (Normalized)')
plt.title('Real EPP vs. Generated EPP (Normalized) with Euclidean Distance')
plt.xlabel('Sample Index')
plt.ylabel('EPP Value')
plt.legend()
plt.grid(True)
plt.show()