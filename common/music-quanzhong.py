import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from brian2 import *
import warnings

# 设置警告过滤器，忽略特定类型的警告
warnings.filterwarnings("ignore", message="Bad key", category=UserWarning)

plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体

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
data = pd.read_excel('music-test111.xlsx', engine='openpyxl')

# 提取特征和目标变量
X = data[['pitch_mean', 'tonnetz_mean', 'rms_mean', 'tempo_mean', 'duration_mean']]
y = data['EPP']

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

# 模拟丘脑和感觉皮层部分
thalamus_output = np.random.uniform(0, 1, size=(n, 2))  # 模拟丘脑输出信号
cortex_input = thalamus_output  # 感觉皮层接收丘脑输出信号并进行处理

# 模拟杏仁核部分
amygdala_input = np.concatenate((cortex_input, thalamus_output), axis=1)  # 杏仁核接收来自感觉皮层和丘脑的输入
excitatory_system_output = np.zeros((n, 1))
inhibitory_system_output = np.zeros((n, 1))

# 模拟兴奋性学习系统
for i in range(n):
    excitatory_system_output[i] = np.sum(amygdala_input[i])  # 简单求和作为模拟

# 模拟抑制性输出系统
for i in range(n):
    inhibitory_system_output[i] = np.sum(amygdala_input[i])  # 简单求和作为模拟

# 模拟眶额皮层部分
orbitofrontal_input = np.concatenate((cortex_input, amygdala_input), axis=1)  # 眶额皮层接收来自其他皮层区域的输入
orbitofrontal_output = np.zeros((n, 1))

# 模拟眶额皮层的抑制性功能
for i in range(n):
    orbitofrontal_output[i] = np.sum(orbitofrontal_input[i])  # 简单求和作为模拟

# 循环神经网络参数
hidden_size = 10
eta = 0.000000000000001
eta_m = 0.00000000000004
eta_o = 0.05  # 眶额皮层学习率
theta = 0.5  # 眶额皮层阈值
rew = 2

# 初始化权重
vi = np.random.uniform(-1, 1, size=(1, 2))
wi = np.random.uniform(-1, 1, size=(1, 2))
Ai = np.zeros((n, 2))
Oi = np.zeros((n, 2))
we = np.random.uniform(-1, 1, size=(2, 2))

# 训练参数
epoch = 100
number_train = round(0.75 * n)
number_test = n - number_train

# 记录每个周期的准确率
accuracies = []

# 训练
for iter in range(epoch):
    for i in range(number_train):
        # 获取输入数据和目标数据
        input_data = rnn_input[i]
        target_data = target[i]

        x, y = input_data[-1], input_data[-2]  # 获取最后一个时间步的两个特征值
        z = target_data[0]  # 目标值

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

        # 更新眶额皮层
        we += eta_m * (Ai[i].reshape(-1, 1) - theta) * Oi[i].reshape(1, -1)

    # 计算当前模型在测试集上的准确率
    if iter == epoch - 1:  # 仅在最后一轮显示准确率
        correct = 0
        for i in range(number_test):
            x, y = rnn_input[number_train + i, -2], rnn_input[number_train + i, -1]

            z = target[number_train + i][0]  # 目标值

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
        print(f'Epoch {iter + 1}, 测试准确率: {accuracy:.2f}%')

        # 归一化预测值
        output_normalized = (output - np.min(output)) / (np.max(output) - np.min(output))

        # 绘制预测值和原始值的对比图
        plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体
        plt.plot(range(number_test), output_normalized[number_train:], label='预测值')
        plt.plot(range(number_test), target[number_train:], label='原始 EPP')
        plt.title('预测值与原始 EPP 对比')
        plt.xlabel('数据索引')
        plt.ylabel('EPP')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 绘制权重热图
        plt.figure(figsize=(12, 4))

        # vi 权重热图
        plt.subplot(1, 3, 1)
        plt.title('vi 权重')
        plt.imshow(vi, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.clim(-2, 2)  # 设置颜色条范围，使不同矩阵的颜色条一致
        plt.xlabel('列')
        plt.ylabel('行')

        # wi 权重热图
        plt.subplot(1, 3, 2)
        plt.title('wi 权重')
        plt.imshow(wi, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.clim(-1, 1)
        plt.xlabel('列')
        plt.ylabel('行')

        # we 权重热图
        plt.subplot(1, 3, 3)
        plt.title('we 权重')
        plt.imshow(we, cmap='coolwarm', aspect='auto')
        plt.colorbar()
        plt.clim(-1, 1)
        plt.xlabel('列')
        plt.ylabel('行')

        plt.tight_layout()
        plt.show()

        # 使用 pickle 保存模型参数
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
