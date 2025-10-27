class RBFNNFaultLocalization:
    def __init__(self, beta):
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


import os
import csv
from configparser import ConfigParser
import re
import torch
import numpy as np
from scipy.linalg import pinv


class RBFNNWithAttention(RBFNNFaultLocalization):
    def __init__(self, beta, func_attention_scale=1.0, hidden_attention_scale=1.0):
        super().__init__(beta)
        self.func_attention_weights = None  # 函数级注意力权重（聚焦稀疏覆盖函数）
        self.hidden_attention_weights = None  # 隐藏层注意力权重（抑制仅匹配成功用例的中心）
        self.func_attention_scale = func_attention_scale  # 函数注意力强度缩放
        self.hidden_attention_scale = hidden_attention_scale  # 隐藏层注意力强度缩放

    def _compute_function_attention(self, C):
        """
        计算函数级注意力权重：被少量测试用例覆盖的函数权重更高
        :param C: 覆盖矩阵 (n, m)，n=测试用例数，m=函数数
        :return: 函数权重 (m,)，值越大表示函数越稀疏（覆盖的测试用例越少）
        """
        # 1. 计算每个函数被多少测试用例覆盖（覆盖频率）
        func_coverage_count = np.sum(C, axis=0)  # (m,)，每个函数的覆盖次数
        # 2. 稀疏度 = 1 / (覆盖次数 + 1e-6)（避免除零），覆盖次数越少，稀疏度越高
        func_sparsity = 1.0 / (func_coverage_count + 1e-6)
        # 3. 归一化权重（确保权重在0~1之间）
        func_weights = func_sparsity / np.max(func_sparsity)
        # 4. 缩放权重（增强区分度）
        return func_weights ** self.func_attention_scale

    def _compute_hidden_attention(self, hidden_centers, C, r):
        """
        计算隐藏层注意力权重：仅匹配成功测试用例的中心权重降低
        :param hidden_centers: 隐藏层中心 (h, m)
        :param C: 覆盖矩阵 (n, m)
        :param r: 测试结果标签 (n, 1)，1=失败，0=成功
        :return: 隐藏层中心权重 (h,)，值越大表示中心与失败用例匹配度越高
        """
        h = len(hidden_centers)
        n = len(r)
        count_l = np.sum(C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)

        # 1. 计算每个中心与测试用例的匹配度（激活值）
        center_activations = []
        for mu in hidden_centers:
            activations = [np.exp(-(self._compute_wbc_distance(c, mu, count_l) **2) / (2 * self.sigma** 2))
                          for c in C]
            center_activations.append(activations)  # (h, n)

        # 2. 区分失败/成功测试用例的激活值
        fail_mask = (r.flatten() == 1)  # 失败用例掩码 (n,)
        success_mask = ~fail_mask       # 成功用例掩码

        # 3. 每个中心的“失败匹配度” = 与失败用例的平均激活值
        #    “成功匹配度” = 与成功用例的平均激活值
        fail_matching = [np.mean(np.array(acts)[fail_mask]) if np.any(fail_mask) else 0.0
                         for acts in center_activations]
        success_matching = [np.mean(np.array(acts)[success_mask]) if np.any(success_mask) else 0.0
                           for acts in center_activations]

        # 4. 权重 = 失败匹配度 / (失败匹配度 + 成功匹配度 + 1e-6)
        #    仅匹配成功用例的中心，权重会趋近于0
        hidden_weights = np.array([
            f / (f + s + 1e-6) for f, s in zip(fail_matching, success_matching)
        ])
        # 5. 缩放权重（增强区分度）
        return hidden_weights ** self.hidden_attention_scale

    def train(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        self.X = X
        self.n, total_cols = X.shape
        self.m = total_cols - 1
        self.C = X[:, :-1]
        self.r = X[:, -1].reshape(-1, 1)

        # 1. 应用函数级注意力：对覆盖矩阵的列（函数）加权
        self.func_attention_weights = self._compute_function_attention(self.C)
        weighted_C = self.C * self.func_attention_weights  # (n, m)，稀疏函数被放大
        self.C = weighted_C  # 替换为加权后的覆盖矩阵

        # 2. 筛选隐藏层中心（基于加权后的覆盖矩阵）
        self.hidden_centers = self._select_hidden_centers()
        h = len(self.hidden_centers)
        if h == 0:
            raise ValueError("未筛选出隐藏层中心，需调整β阈值")

        # 3. 计算全局sigma
        self.sigma = self._compute_global_sigma(self.hidden_centers)
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)

        # 4. 计算隐藏层注意力权重（抑制仅匹配成功用例的中心）
        self.hidden_attention_weights = self._compute_hidden_attention(
            self.hidden_centers, self.C, self.r
        )

        # 5. 计算激活矩阵A（融入隐藏层注意力）
        A = []
        for c in self.C:
            activations = [np.exp(-(self._compute_wbc_distance(c, mu, count_l) **2) / (2 * self.sigma** 2))
                          for mu in self.hidden_centers]
            # 用隐藏层权重加权激活值：仅匹配成功用例的中心激活值被削弱
            weighted_activations = np.array(activations) * self.hidden_attention_weights
            A.append(weighted_activations)
        self.A = np.array(A)

        # 6. 求解输出层权重
        self.weights = pinv(self.A) @ self.r

    def compute_suspiciousness(self):
        if self.weights is None:
            raise ValueError("RBFNN未训练，无法计算可疑度")
        h = len(self.hidden_centers)
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        suspiciousness = []
        for j in range(self.m):
            # 虚拟覆盖向量：仅激活第j个函数
            virtual_c = np.zeros(self.m)
            virtual_c[j] = 1
            # 应用函数级注意力（稀疏函数的虚拟向量被放大）
            virtual_c_weighted = virtual_c * self.func_attention_weights

            # 计算虚拟激活值（融入隐藏层注意力）
            virtual_activations = [np.exp(-(self._compute_wbc_distance(virtual_c_weighted, mu, count_l) **2) / (2 * self.sigma** 2))
                                 for mu in self.hidden_centers]
            virtual_activations = np.array(virtual_activations) * self.hidden_attention_weights
            virtual_activations = virtual_activations.reshape(1, -1)

            # 计算可疑度
            susp = (virtual_activations @ self.weights)[0][0]
            suspiciousness.append(susp)
        return np.array(suspiciousness)


def fileRank(bugId, rev, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')
    resultFile = baseResultDir + bugId + '/NNresultFile_file_with_attention.csv'
    if not os.path.exists(baseResultDir + bugId):
        os.mkdir(baseResultDir + bugId)

    matrix = []
    methodmap = dict()
    # 读取失败测试用例信息
    with open(baseInfoDir + bugId + '/failcov/method_info.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            items = line.strip().split(',')
            if len(items) < 2:
                continue
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            methodname = items[1]
            key = f"{filename},{methodname}"
            if key not in methodmap:
                methodmap[key] = len(methodmap)
        vector = torch.zeros(len(methodmap) + 1)
        vector[-1] = 1
        for v in methodmap.values():
            vector[v] = 1
        matrix.append(vector)

    # 读取剩余失败测试用例
    failcovDir = basePassDir + bugId + '/failcov'
    if os.path.exists(failcovDir):
        for dir in os.listdir(failcovDir):
            vector = torch.zeros(len(methodmap) + 1)
            vector[-1] = 1
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

    # 读取成功测试用例
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
                    vector[methodmap[key]] = 1
        matrix.append(vector)

    cov_matrix = torch.stack(matrix)
    shuffle_indices = torch.randperm(cov_matrix.size(0))
    shuffled_matrix = cov_matrix[shuffle_indices]
    count = (cov_matrix[:, -1] == 1).sum().item()
    print(f"覆盖矩阵形状: {cov_matrix.shape}")
    print(f"失败测试用例数: {count}")
    print(f"通过测试用例数: {len(cov_matrix) - count}")

    if cov_matrix is not None and cov_matrix.shape[0] > 1:
        # 使用带注意力的RBFNN，可调整参数增强效果
        rbfnn = RBFNNWithAttention(
            beta=0.05,
            func_attention_scale=5.0,  # 放大函数稀疏度的影响（>1增强，<1减弱）
            hidden_attention_scale=5.0  # 放大隐藏层权重的影响
        )
        rbfnn.train(cov_matrix)
        function_suspiciousness = rbfnn.compute_suspiciousness()
        ranked_functions = rbfnn.rank_functions()

        index_to_method = {v: k for k, v in methodmap.items()}
        file_susp_dict = dict()
        for func_idx in range(len(function_suspiciousness)):
            method_key = index_to_method.get(func_idx)
            if not method_key:
                continue
            filename = method_key.split(',', 1)[0]
            susp = function_suspiciousness[func_idx]
            if filename not in file_susp_dict:
                file_susp_dict[filename] = []
            file_susp_dict[filename].append(susp)

        file_avg_susp = []
        for filename, susp_list in file_susp_dict.items():
            avg_susp = np.mean(susp_list)
            file_avg_susp.append((filename, avg_susp))
        file_avg_susp.sort(key=lambda x: x[1], reverse=True)

        with open(resultFile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'File', 'Score'])
            for rank, (filename, avg_susp) in enumerate(file_avg_susp, 1):
                writer.writerow([rank, filename, round(avg_susp, 6)])

        print(f"带自注意力的文件可疑度排序已输出到：{resultFile}")

    else:
        print("错误: 数据不足，无法训练模型")