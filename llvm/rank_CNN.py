import os
import csv
from configparser import ConfigParser
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# ==========================================
# 1. CNN-FL 模型定义
# ==========================================
class CNN_FL(nn.Module):
    def __init__(self, num_statements):
        super(CNN_FL, self).__init__()

        # 论文 Section III Architecture 描述[cite: 174]:
        # Input -> Conv1 -> ReLU -> Pool1 -> Conv2 -> ReLU -> Pool2 -> FCs -> Output

        # 卷积层 1: 32 kernels, kernel size 10 [cite: 187, 188]
        # 输入通道为 1 (单条覆盖率向量视为序列)
        # padding=5 是为了保持一定的序列长度，防止卷积后维度消失过快
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=10, padding=5)
        self.relu1 = nn.ReLU()  # [cite: 191]
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # Max Pooling [cite: 216]

        # 卷积层 2: 64 kernels, kernel size 10 [cite: 187]
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=10, padding=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        # 动态计算卷积层输出后的扁平化维度 (适配不同的 num_statements)
        self._to_linear = None
        self._check_flatten_size(num_statements)

        # 全连接层: 论文指出有 3 层，每层 1024 个节点 [cite: 288]
        self.fc1 = nn.Linear(self._to_linear, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 1024)

        # 输出层: 1 个节点，使用 Sigmoid 激活 [cite: 289]
        self.fc_out = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def _check_flatten_size(self, input_len):
        """辅助函数：通过一次假的前向传播来计算 Linear 层的输入维度"""
        with torch.no_grad():
            # Batch=1, Channel=1, Length=N
            x = torch.zeros(1, 1, input_len)
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            self._to_linear = x.shape[1] * x.shape[2]

    def forward(self, x):
        # 输入形状变换: (Batch, N) -> (Batch, 1, N) 以适配 Conv1d
        x = x.unsqueeze(1)

        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))

        # 展平 (Flatten)
        x = x.view(-1, self._to_linear)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        output = self.sigmoid(self.fc_out(x))
        return output


# ==========================================
# 2. 主逻辑函数
# ==========================================
def fileRank_CNN(bugId, rev, configFile):
    # 配置读取
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')

    # 输出文件路径
    resultFile = os.path.join(baseResultDir, bugId, 'result_cnn.csv')
    if not os.path.exists(os.path.join(baseResultDir, bugId)):
        os.mkdir(os.path.join(baseResultDir, bugId))

    matrix = []
    methodmap = dict()

    # -----------------------------
    # 数据读取部分 (保持原有逻辑)
    # -----------------------------

    # 读取失败测试用例信息
    fail_info_path = baseInfoDir + bugId + '/failcov/method_info.txt'
    if os.path.exists(fail_info_path):
        with open(fail_info_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                if len(items) < 2: continue
                filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                methodname = items[1]
                key = f"{filename},{methodname}"
                if key not in methodmap:
                    methodmap[key] = len(methodmap)

            # 填充向量
            vector = torch.zeros(len(methodmap) + 1)
            vector[-1] = 1  # 失败标签
            for line in lines:
                items = line.strip().split(',')
                if len(items) < 2: continue
                filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                methodname = items[1]
                key = f"{filename},{methodname}"
                if key in methodmap:
                    vector[methodmap[key]] = 1
            matrix.append(vector)

    # 读取成功测试用例
    passcovDir = basePassDir + bugId + '/passcov/'
    cnt = 0
    if os.path.exists(passcovDir):
        for dir_name in os.listdir(passcovDir):
            vector = torch.zeros(len(methodmap) + 1)
            # 成功标签默认为0 (vector[-1]初始化为0)

            method_file = os.path.join(passcovDir, dir_name, 'method_info.txt')
            if not os.path.exists(method_file): continue

            with open(method_file, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split(',')
                    if len(items) < 2: continue
                    filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                    methodname = items[1]
                    key = f"{filename},{methodname}"
                    if key in methodmap:
                        vector[methodmap[key]] = 1
            matrix.append(vector)
            cnt += 1

    # 读取剩余失败测试用例 (Balance data or extra fails)
    # failcovDir = basePassDir + bugId + '/failcov'
    # i = 0
    # if os.path.exists(failcovDir):
    #     for dir_name in os.listdir(failcovDir):
    #         if i >= cnt - 1: break  # 简单平衡
    #         vector = torch.zeros(len(methodmap) + 1)
    #         vector[-1] = 1  # 失败标签
    #
    #         method_file = os.path.join(failcovDir, dir_name, 'method_info.txt')
    #         if not os.path.exists(method_file): continue
    #
    #         with open(method_file, 'r') as f:
    #             lines = f.readlines()
    #             for line in lines:
    #                 items = line.strip().split(',')
    #                 if len(items) < 2: continue
    #                 filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
    #                 methodname = items[1]
    #                 key = f"{filename},{methodname}"
    #                 if key in methodmap:
    #                     vector[methodmap[key]] = 1
    #         matrix.append(vector)
    #         i += 1

    # -----------------------------
    # 数据预处理与 CNN 训练
    # -----------------------------

    # 将列表转换为 Tensor
    if not matrix:
        print("Error: Matrix is empty.")
        return

    full_data = torch.stack(matrix)

    # 划分 X (特征/覆盖率) 和 y (标签/结果)
    # X: [M, N] (M个测试用例, N个方法)
    X_data = full_data[:, :-1]
    # y: [M, 1]
    y_data = full_data[:, -1].unsqueeze(1)

    num_statements = X_data.shape[1]
    print(f"Data loaded: {len(X_data)} test cases, {num_statements} methods.")

    # 检查是否可以使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建 DataLoader, Batch Size = 10 [cite: 312]
    BATCH_SIZE = 10
    dataset = TensorDataset(X_data, y_data)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化 CNN 模型
    model = CNN_FL(num_statements).to(device)

    # 损失函数和优化器
    # 论文使用 SGD [cite: 155]
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    # 训练模型
    EPOCHS = 200  # 这里的轮数可以根据收敛情况调整
    print("Starting CNN Training...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 简单的动态学习率调整 (模拟论文 Eq. 4 的衰减思路 [cite: 306])
        if (epoch + 1) % 5 == 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.98
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(dataloader):.4f}")

    # -----------------------------
    # 故障定位 (Virtual Test Cases)
    # -----------------------------
    # 论文核心步骤：构建虚拟测试用例集，每个用例只覆盖一个语句

    print("Calculating Suspiciousness using Virtual Test Cases...")
    model.eval()
    suspiciousness_scores = {}

    # 构造对角矩阵 (Identity Matrix)，对角线为1，其余为0
    # 这代表 N 个虚拟测试用例，第 i 个用例仅覆盖第 i 个方法
    virtual_test_suite = torch.eye(num_statements).to(device)

    index_to_method = {v: k for k, v in methodmap.items()}

    with torch.no_grad():
        # 逐个输入虚拟测试用例，避免显存溢出
        for i in range(num_statements):
            # 构造单个虚拟测试用例输入: shape (1, N)
            v_input = virtual_test_suite[i].unsqueeze(0)

            # 模型输出即为该语句导致错误的概率 (怀疑度) [cite: 159, 161]
            prob = model(v_input).item()

            method_key = index_to_method.get(i)
            if method_key:
                suspiciousness_scores[method_key] = prob

    # -----------------------------
    # 结果聚合与输出
    # -----------------------------
    print("Aggregating File-level Scores...")

    file_susp_dict = {}

    # 遍历所有方法的怀疑度
    for method_key, score in suspiciousness_scores.items():
        # method_key 格式 "filename,methodname"
        filename = method_key.split(',')[0]

        if filename not in file_susp_dict:
            file_susp_dict[filename] = []
        file_susp_dict[filename].append(score)

    # 计算文件平均怀疑度
    file_final_ranks = []
    for filename, scores in file_susp_dict.items():
        if len(scores) > 0:
            avg_score = sum(scores) / len(scores)
            file_final_ranks.append((filename, avg_score))

    # 按照分数降序排序 (Rank, Filename, Score)
    file_final_ranks.sort(key=lambda x: x[1], reverse=True)

    # 写入 CSV
    try:
        with open(resultFile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'Filename', 'Score'])
            for rank, (filename, score) in enumerate(file_final_ranks, 1):
                writer.writerow([rank, filename, f"{score:.6f}"])

        print(f"CNN-FL Result saved to: {resultFile}")

    except Exception as e:
        print(f"Error saving CSV: {e}")

# 如果需要独立运行，可以在这里调用 fileRank_CNN
# fileRank_CNN("bugId_example", "rev_example", "config.ini")