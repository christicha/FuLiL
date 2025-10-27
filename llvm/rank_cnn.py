import os
import csv
from configparser import ConfigParser
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CNNFaultLocalization:
    def __init__(self, lr=1e-3, weight_decay=1e-4, max_epochs=1000, tol=1e-6):
        """
        初始化CNN故障定位模型（遵循论文设计）
        :param lr: Adam优化器学习率（论文隐含使用，默认1e-3）
        :param weight_decay: L2正则化系数（论文提及防止过拟合，默认1e-4）
        :param max_epochs: 最大训练迭代次数（论文未指定，默认1000次）
        :param tol: 损失收敛阈值（损失小于tol时停止训练，默认1e-6）
        """
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.tol = tol
        self.conv = None  # 卷积层（核心组件，1×N过滤器）
        self.optimizer = None  # Adam优化器
        self.criterion = nn.MSELoss()  # MSE损失函数（论文指定）
        self.m = None  # 函数数量（对应论文中N，即覆盖矩阵列数）
        self.n = None  # 测试用例数量（对应论文中M，即覆盖矩阵行数）

    def _init_conv_layer(self):
        """
        动态初始化卷积层（过滤器大小1×m，与函数数一致，论文3.1节）
        """
        # 卷积层配置：输入通道1，输出通道1，过滤器1×m，步长1
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=(1, self.m),
            stride=1,
            padding=0
        )
        # 过滤器权重随机均匀初始化（论文隐含要求，参考深度学习常规操作）
        nn.init.uniform_(self.conv.weight, a=-0.1, b=0.1)
        nn.init.zeros_(self.conv.bias)
        # 初始化Adam优化器（含L2正则化，论文3.1节）
        self.optimizer = optim.Adam(
            self.conv.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

    def train(self, X):
        """
        训练CNN模型（遵循论文3.1节训练流程）
        :param X: 输入矩阵（shape=[n, m+1]），格式为[函数覆盖特征..., 标签]
        """
        # 1. 数据预处理：分离函数覆盖向量（C）与标签（r），转换为Torch张量
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32)
        self.n, total_cols = X.shape
        self.m = total_cols - 1  # 函数数量m = 总列数-1（最后一列为标签）
        C = X[:, :-1]  # 函数覆盖矩阵（shape=[n, m]）
        r = X[:, -1].reshape(-1, 1)  # 测试结果标签（shape=[n, 1]，0=通过，1=失败）

        # 2. 调整输入形状为CNN要求格式：[batch_size, channels, height, width]
        # 论文中输入为M×N矩阵，对应此处[ n, 1, 1, m ]（height=1，width=m，通道数1）
        C = C.reshape(self.n, 1, 1, self.m)

        # 3. 初始化卷积层与优化器
        self._init_conv_layer()

        # 4. 迭代训练（论文未指定迭代次数，用最大次数+收敛阈值控制）
        prev_loss = float('inf')
        for epoch in range(self.max_epochs):
            # 前向传播：计算预测值（测试用例失败概率）
            self.conv.train()
            y_pred = self.conv(C).squeeze(-1).squeeze(-1).reshape(-1, 1)  # 输出shape=[n, 1]

            # 计算损失（MSE）
            loss = self.criterion(y_pred, r)

            # 反向传播与权重更新
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 收敛判断：损失变化小于tol时停止训练
            if abs(prev_loss - loss.item()) < self.tol:
                print(f"训练收敛，迭代次数：{epoch+1}，最终损失：{loss.item():.6f}")
                break
            prev_loss = loss.item()

        else:
            print(f"达到最大迭代次数{self.max_epochs}，训练停止，最终损失：{prev_loss:.6f}")

    def compute_suspiciousness(self):
        """
        计算函数可疑度（遵循论文3.1节测试流程：输入单位矩阵）
        :return: suspiciousness（shape=[m]），值越大函数越可能含故障
        """
        if self.conv is None:
            raise ValueError("CNN未训练，无法计算可疑度")

        # 1. 生成测试输入：n×n单位矩阵（每行仅1个函数被覆盖，论文3.1节）
        # 此处单位矩阵维度为[m, m]（m=函数数），对应每个函数的虚拟测试用例
        identity_matrix = torch.eye(self.m, dtype=torch.float32)  # shape=[m, m]
        # 调整为CNN输入格式：[m, 1, 1, m]
        identity_matrix = identity_matrix.reshape(self.m, 1, 1, self.m)

        # 2. 前向传播计算可疑度（输出为函数故障可能性）
        self.conv.eval()
        with torch.no_grad():  # 测试阶段不更新权重
            suspiciousness = self.conv(identity_matrix).squeeze().numpy()  # shape=[m]

        # 3. （可选）归一化到0-1范围（论文6.1节提及CNN输出范围-1~1，归一化后便于排序）
        min_susp = suspiciousness.min()
        max_susp = suspiciousness.max()
        if max_susp > min_susp:
            suspiciousness = (suspiciousness - min_susp) / (max_susp - min_susp)
        return suspiciousness

    def rank_functions(self):
        """
        按可疑度降序排序函数（与原RBFNN接口一致，确保fileRank兼容）
        :return: ranked_indices（shape=[m]），索引0为最可疑函数
        """
        suspiciousness = self.compute_suspiciousness()
        return np.argsort(-suspiciousness)


def fileRank(bugId, rev, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')
    resultFile = baseResultDir + bugId + '/CNNresultFile_file.csv'  # 输出CSV路径（修改为CNN标识）
    if not os.path.exists(baseResultDir + bugId):
        os.mkdir(baseResultDir + bugId)

    # 1. 构建函数覆盖矩阵与methodmap（函数→索引映射，逻辑与原代码一致）
    matrix = []
    methodmap = dict()  # 键："文件名,函数名"，值：函数索引
    # 读取失败测试用例的函数信息，初始化methodmap
    with open(baseInfoDir + bugId + '/failcov/method_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            if len(items) < 2:
                continue  # 跳过格式错误行
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            methodname = items[1]
            key = f"{filename},{methodname}"
            if key not in methodmap:
                methodmap[key] = len(methodmap)
        # 添加首个失败测试用例覆盖向量（标签1）
        vector = torch.zeros(len(methodmap) + 1)
        vector[-1] = 1
        for v in methodmap.values():
            vector[v] = 1
        matrix.append(vector)

    # 读取剩余失败测试用例（新增逻辑，原代码已包含）
    failcovDir = basePassDir + bugId + '/failcov'
    if os.path.exists(failcovDir):
        for dir in os.listdir(failcovDir):
            vector = torch.zeros(len(methodmap) + 1)
            vector[-1] = 1  # 失败测试用例标签为1
            with open(os.path.join(failcovDir, dir, 'method_info.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split(',')
                    if len(items) < 2:
                        continue
                    filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                    methodname = items[1]
                    key = f"{filename},{methodname}"
                    if key in methodmap:
                        vector[methodmap[key]] = 1
            matrix.append(vector)

    # 读取成功测试用例（标签0，逻辑与原代码一致）
    passcovDir = basePassDir + bugId + '/passcov/'
    for dir in os.listdir(passcovDir):
        vector = torch.zeros(len(methodmap) + 1)  # 成功测试用例标签默认0
        with open(os.path.join(passcovDir, dir, 'method_info.txt'), 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.strip().split(',')
                if len(items) < 2:
                    continue
                filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                methodname = items[1]
                key = f"{filename},{methodname}"
                if key in methodmap:
                    vector[methodmap[key]] = 1
        matrix.append(vector)

    # 2. 构建覆盖矩阵并打乱（逻辑与原代码一致）
    cov_matrix = torch.stack(matrix)
    shuffle_indices = torch.randperm(cov_matrix.size(0))
    shuffled_matrix = cov_matrix[shuffle_indices]  # 打乱测试用例顺序
    count = (cov_matrix[:, -1] == 1).sum().item()  # 统计失败测试用例数
    print(f"覆盖矩阵形状: {cov_matrix.shape}")
    print(f"失败测试用例数: {count}")
    print(f"通过测试用例数: {len(cov_matrix) - count}")

    # 3. CNN模型训练与结果计算（替换原RBFNN逻辑）
    if cov_matrix is not None and cov_matrix.shape[0] > 1:
        # 初始化CNN（参数可根据论文实验调整，此处用默认值）
        cnn = CNNFaultLocalization(
            lr=1e-3,        # 学习率（论文未指定，参考常规Adam参数）
            weight_decay=1e-4,  # L2正则化系数（论文3.1节提及）
            max_epochs=1000,    # 最大迭代次数
            tol=1e-6            # 收敛阈值
        )
        cnn.train(cov_matrix)  # 训练CNN
        function_suspiciousness = cnn.compute_suspiciousness()  # 函数可疑度
        ranked_functions = cnn.rank_functions()  # 函数排序

        # 4. 计算文件平均可疑度（逻辑与原代码完全一致）
        index_to_method = {v: k for k, v in methodmap.items()}  # 索引→"文件名,函数名"映射
        file_susp_dict = dict()  # 键：文件名，值：函数可疑度列表
        for func_idx in range(len(function_suspiciousness)):
            method_key = index_to_method.get(func_idx)
            if not method_key:
                continue
            filename = method_key.split(',', 1)[0]  # 提取文件名（按首个逗号分割）
            susp = function_suspiciousness[func_idx]
            if filename not in file_susp_dict:
                file_susp_dict[filename] = []
            file_susp_dict[filename].append(susp)

        # 计算文件平均可疑度并排序
        file_avg_susp = []
        for filename, susp_list in file_susp_dict.items():
            avg_susp = np.mean(susp_list)
            file_avg_susp.append((filename, avg_susp))
        file_avg_susp.sort(key=lambda x: x[1], reverse=True)  # 降序排序

        # 5. 输出到CSV（逻辑与原代码一致）
        with open(resultFile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'File', 'Score'])  # 表头
            for rank, (filename, avg_susp) in enumerate(file_avg_susp, 1):
                writer.writerow([rank, filename, round(avg_susp, 6)])

        print(f"文件可疑度排序已输出到：{resultFile}")

    else:
        print("错误: 数据不足，无法训练模型")