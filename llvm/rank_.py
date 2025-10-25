import os
import csv  # 新增：用于CSV文件操作
from configparser import ConfigParser
import re
import torch
import numpy as np
from scipy.linalg import pinv


class RBFNNFaultLocalization:
    # （此类代码与之前一致，省略重复部分，核心逻辑不变）
    def __init__(self, beta=0.1):
        self.beta = beta
        self.hidden_centers = None
        self.sigma = None
        self.weights = None
        self.m = None
        self.n = None
        self.X = None
        self.C = None
        self.r = None

    def _compute_wbc_distance(self, x, mu, count_l):
        diff = np.abs(x - mu)
        weighted_diff = diff / count_l
        wbc_dist = np.sqrt(np.mean(weighted_diff))
        return wbc_dist

    def _select_hidden_centers(self):
        if self.n == 0:
            raise ValueError("无测试用例数据，无法选择隐藏层中心")
        hidden_centers = [self.C[0]]
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        for c in self.C[1:]:
            min_dist = min(self._compute_wbc_distance(c, mu, count_l) for mu in hidden_centers)
            if min_dist >= self.beta:
                hidden_centers.append(c)
        return np.array(hidden_centers)

    def _compute_global_sigma(self, hidden_centers):
        h = len(hidden_centers)
        if h == 1:
            return 0.1
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        nearest_distances = []
        for i in range(h):
            mu_i = hidden_centers[i]
            dists = [self._compute_wbc_distance(mu_i, hidden_centers[j], count_l) for j in range(h) if j != i]
            nearest_distances.append(min(dists))
        return np.mean(nearest_distances)

    def train(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        self.X = X
        self.n, total_cols = X.shape
        self.m = total_cols - 1
        self.C = X[:, :-1]
        self.r = X[:, -1].reshape(-1, 1)
        self.hidden_centers = self._select_hidden_centers()
        h = len(self.hidden_centers)
        if h == 0:
            raise ValueError("未筛选出隐藏层中心，需调整β阈值")
        self.sigma = self._compute_global_sigma(self.hidden_centers)
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        A = []
        for c in self.C:
            activations = [np.exp(-(self._compute_wbc_distance(c, mu, count_l) **2) / (2 * self.sigma** 2)) for mu in self.hidden_centers]
            A.append(activations)
        self.A = np.array(A)
        self.weights = pinv(self.A) @ self.r

    def compute_suspiciousness(self):
        if self.weights is None:
            raise ValueError("RBFNN未训练，无法计算可疑度")
        h = len(self.hidden_centers)
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        suspiciousness = []
        for j in range(self.m):
            virtual_c = np.zeros(self.m)
            virtual_c[j] = 1
            virtual_activations = [np.exp(-(self._compute_wbc_distance(virtual_c, mu, count_l) **2) / (2 * self.sigma** 2)) for mu in self.hidden_centers]
            virtual_activations = np.array(virtual_activations).reshape(1, -1)
            susp = (virtual_activations @ self.weights)[0][0]
            suspiciousness.append(susp)
        return np.array(suspiciousness)

    def rank_functions(self):
        suspiciousness = self.compute_suspiciousness()
        return np.argsort(-suspiciousness)


def fileRank(bugId, rev, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')
    resultFile = baseResultDir + bugId + '/NNresultFile_file.csv'  # 输出的CSV路径
    if not os.path.exists(baseResultDir + bugId):
        os.mkdir(baseResultDir + bugId)

    # 1. 构建函数覆盖矩阵和methodmap（函数到索引的映射）
    matrix = []
    methodmap = dict()  # 键："文件名,函数名"，值：函数索引
    # 读取失败测试用例的函数信息，构建methodmap
    with open(baseInfoDir + bugId + '/failcov/method_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')  # 分割文件名和函数名（假设无多余逗号）
            if len(items) < 2:
                continue  # 跳过格式错误的行
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            methodname = items[1]
            key = f"{filename},{methodname}"
            if key not in methodmap:
                methodmap[key] = len(methodmap)  # 分配唯一索引
        # 添加失败测试用例的覆盖向量（标签为1）
        vector = torch.zeros(len(methodmap) + 1)
        vector[-1] = 1  # 标签：失败测试用例为1
        for v in methodmap.values():
            vector[v] = 1
        matrix.append(vector)

    # 读取成功测试用例的函数信息，构建覆盖向量（标签为0）
    passcovDir = basePassDir + bugId + '/passcov/'
    for dir in os.listdir(passcovDir):
        vector = torch.zeros(len(methodmap) + 1)
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
                    vector[methodmap[key]] = 1  # 标记函数被覆盖
        matrix.append(vector)

    # 2. 构建覆盖矩阵并训练RBFNN
    cov_matrix = torch.stack(matrix)
    count = (cov_matrix[:, -1] == 1).sum().item()
    print(f"覆盖矩阵形状: {cov_matrix.shape}")
    print(f"失败测试用例数: {count}")
    print(f"通过测试用例数: {len(cov_matrix) - count}")

    if cov_matrix is not None and cov_matrix.shape[0] > 1:
        rbfnn = RBFNNFaultLocalization(beta=0.1)
        rbfnn.train(cov_matrix)
        function_suspiciousness = rbfnn.compute_suspiciousness()  # 每个函数的可疑度
        ranked_functions = rbfnn.rank_functions()  # 函数索引排序

        # 3. 计算文件的平均可疑度
        # 3.1 建立函数索引到“文件名,函数名”的映射（反转methodmap）
        index_to_method = {v: k for k, v in methodmap.items()}  # 键：函数索引，值："文件名,函数名"
        # 3.2 按文件分组函数可疑度
        file_susp_dict = dict()  # 键：文件名，值：[该文件下所有函数的可疑度列表]
        for func_idx in range(len(function_suspiciousness)):
            method_key = index_to_method.get(func_idx)
            if not method_key:
                continue
            # 分割文件名（处理可能的逗号，只按第一个逗号分割）
            filename = method_key.split(',', 1)[0]  # 取逗号前的部分作为文件名
            susp = function_suspiciousness[func_idx]
            # 添加到文件对应的列表中
            if filename not in file_susp_dict:
                file_susp_dict[filename] = []
            file_susp_dict[filename].append(susp)

        # 3.3 计算每个文件的平均可疑度
        file_avg_susp = []
        for filename, susp_list in file_susp_dict.items():
            avg_susp = np.mean(susp_list)  # 求平均值
            file_avg_susp.append((filename, avg_susp))

        # 3.4 按平均可疑度降序排序（从高到低）
        file_avg_susp.sort(key=lambda x: x[1], reverse=True)

        # 4. 输出到CSV文件（表头：Rank, File, Score）
        with open(resultFile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'File', 'Score'])  # 写入表头
            for rank, (filename, avg_susp) in enumerate(file_avg_susp, 1):  # Rank从1开始
                writer.writerow([rank, filename, round(avg_susp, 6)])  # 保留6位小数

        print(f"文件可疑度排序已输出到：{resultFile}")
        print("前5个高可疑度文件：")
        for i in range(min(5, len(file_avg_susp))):
            print(f"Rank {i+1}: {file_avg_susp[i][0]} (Score: {file_avg_susp[i][1]:.6f})")

    else:
        print("错误: 数据不足，无法训练模型")