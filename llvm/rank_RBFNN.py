import os
import csv
from configparser import ConfigParser
import re
import torch
import numpy as np
from scipy.linalg import pinv


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

        print(f"最终隐藏层中心数量：{len(hidden_centers)}")
        return np.array(hidden_centers)

    def _compute_global_sigma(self, hidden_centers):
        h = len(hidden_centers)
        if h == 1:
            return 1.2
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        nearest_distances = []
        for i in range(h):
            mu_i = hidden_centers[i]
            dists = [self._compute_wbc_distance(mu_i, hidden_centers[j], count_l) for j in range(h) if j != i]
            nearest_distances.append(min(dists))
        sigma = np.mean(nearest_distances)
        min_sigma = 0.2 + h / 100
        return max(sigma, min_sigma) * 2

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
            activations = [np.exp(-(self._compute_wbc_distance(c, mu, count_l) ** 2) / (2 * self.sigma ** 2)) for mu in
                           self.hidden_centers]
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
            virtual_activations = [
                np.exp(-(self._compute_wbc_distance(virtual_c, mu, count_l) ** 2) / (2 * self.sigma ** 2)) for mu in
                self.hidden_centers]
            virtual_activations = np.array(virtual_activations).reshape(1, -1)
            print(f"函数{j}虚拟激活值：", virtual_activations)
            susp = (virtual_activations @ self.weights)[0][0]
            suspiciousness.append(susp)

        return np.array(suspiciousness)

    def rank_functions(self):
        suspiciousness = self.compute_suspiciousness()
        return np.argsort(-suspiciousness)


class RBFNNWithAttention(RBFNNFaultLocalization):
    def __init__(self, beta, func_attention_scale, hidden_attention_scale, noise_suppress_scale, fault_corr_scale):
        super().__init__(beta)
        self.func_attention_weights = None  # 函数级权重（聚焦稀疏函数）
        self.hidden_attention_weights = None  # 隐藏层权重（抑制仅匹配成功用例）
        self.testcase_attention_weights = None  # 测试用例权重（抑制噪声用例）
        self.fault_corr_weights = None  # 故障关联权重（仅失败覆盖函数加权）
        self.func_attention_scale = func_attention_scale
        self.hidden_attention_scale = hidden_attention_scale
        self.noise_suppress_scale = noise_suppress_scale
        self.fault_corr_scale = fault_corr_scale  # 故障关联权重缩放强度

    def _compute_function_attention(self, C):
        """函数级注意力：稀疏覆盖函数权重更高"""
        func_coverage_count = np.sum(C, axis=0)  # 每个函数被多少测试用例覆盖
        func_sparsity = 1.0 / (func_coverage_count + 1e-6)  # 覆盖越少，稀疏度越高
        func_weights = func_sparsity / np.max(func_sparsity)  # 归一化
        return func_weights ** self.func_attention_scale

    def _compute_fault_correlation_attention(self, C, r):
        """故障关联注意力：仅失败用例覆盖的函数权重增加"""
        n, m = C.shape
        fail_mask = (r.flatten() == 1)  # 失败用例掩码
        success_mask = ~fail_mask  # 成功用例掩码

        # 1. 计算每个函数被失败/成功用例覆盖的次数
        func_fail_count = np.sum(C[fail_mask], axis=0)  # 被失败用例覆盖的次数
        func_success_count = np.sum(C[success_mask], axis=0)  # 被成功用例覆盖的次数

        # 2. 故障关联度 = 失败覆盖次数 / (总覆盖次数 + 1e-6)
        # 仅失败覆盖：关联度=1；仅成功覆盖：关联度=0；混合覆盖：关联度=失败占比
        total_count = func_fail_count + func_success_count + 1e-6
        fault_correlation = func_fail_count / total_count

        # 3. 缩放权重（增强区分度），仅失败覆盖函数权重被放大
        fault_corr_weights = fault_correlation ** self.fault_corr_scale
        return fault_corr_weights

    def _compute_testcase_difference(self, c1, c2, count_l):
        """加权位比较差异度：对稀疏函数的差异赋予更高权重"""
        diff = np.abs(c1 - c2)  # 位比较：0表示相同，1表示不同
        weighted_diff = diff / count_l  # 稀疏函数（count_l小）的差异被放大
        return np.sqrt(np.mean(weighted_diff))  # 加权差异度

    def _compute_testcase_attention(self, C):
        """测试用例级注意力：基于加权位比较抑制噪声用例"""
        n, m = C.shape
        if n <= 1:
            return np.ones(n)  # 测试用例太少时，不抑制任何用例

        # 计算每个函数的覆盖次数（与隐藏层聚类逻辑一致）
        count_l = np.sum(C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)  # 避免除零

        # 1. 计算每个测试用例与其他所有用例的平均加权差异度
        pairwise_diffs = []
        for i in range(n):
            # 用例i与其他所有用例的加权差异度
            diffs = [self._compute_testcase_difference(C[i], C[j], count_l)
                     for j in range(n) if j != i]
            avg_diff = np.mean(diffs)  # 平均差异度越大，用例越"特殊"（噪声可能性高）
            pairwise_diffs.append(avg_diff)

        # 2. 基于差异度分布识别噪声：差异度 > 均值 + 0.5*标准差 的用例视为噪声
        diff_mean = np.mean(pairwise_diffs)
        diff_std = np.std(pairwise_diffs)
        noise_threshold = diff_mean + 0.5 * diff_std

        # 3. 噪声用例权重降低，正常用例权重升高
        testcase_weights = []
        for diff in pairwise_diffs:
            if diff > noise_threshold:
                # 噪声用例：权重随差异度增加而指数衰减
                weight = np.exp(-(diff - noise_threshold) * self.noise_suppress_scale)
            else:
                # 正常用例：权重为1（不衰减）
                weight = 1.0
            testcase_weights.append(weight)

        # 4. 归一化权重（确保权重和为n，避免整体缩放）
        testcase_weights = np.array(testcase_weights)
        testcase_weights = testcase_weights * (n / np.sum(testcase_weights))
        return testcase_weights

    def _compute_hidden_attention(self, hidden_centers, C, r):
        """隐藏层注意力：抑制仅匹配成功用例的中心"""
        h = len(hidden_centers)
        n = len(r)
        count_l = np.sum(C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)

        # 计算每个中心与测试用例的激活值
        center_activations = []
        for mu in hidden_centers:
            activations = [np.exp(-(self._compute_wbc_distance(c, mu, count_l) ** 2) / (2 * self.sigma ** 2))
                           for c in C]
            center_activations.append(activations)  # (h, n)

        # 区分失败/成功用例的激活值
        fail_mask = (r.flatten() == 1)
        success_mask = ~fail_mask

        # 计算失败/成功匹配度
        fail_matching = [np.mean(np.array(acts)[fail_mask]) if np.any(fail_mask) else 0.0
                         for acts in center_activations]
        success_matching = [np.mean(np.array(acts)[success_mask]) if np.any(success_mask) else 0.0
                            for acts in center_activations]

        # 权重 = 失败匹配度 / (失败+成功匹配度)，抑制仅匹配成功的中心
        hidden_weights = np.array([
            f / (f + s + 1e-6) for f, s in zip(fail_matching, success_matching)
        ])
        return hidden_weights ** self.hidden_attention_scale

    def train(self, X):
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        self.X = X
        self.n, total_cols = X.shape
        self.m = total_cols - 1
        self.C = X[:, :-1]
        self.r = X[:, -1].reshape(-1, 1)

        # 1. 测试用例级注意力：基于加权位比较抑制噪声用例（对覆盖矩阵行加权）
        self.testcase_attention_weights = self._compute_testcase_attention(self.C)
        weighted_C = self.C * self.testcase_attention_weights.reshape(-1, 1)  # (n, m)

        # 2. 函数级注意力：聚焦稀疏函数（对覆盖矩阵列加权）
        self.func_attention_weights = self._compute_function_attention(weighted_C)

        # 新增：3. 故障关联注意力：仅失败用例覆盖的函数加权
        self.fault_corr_weights = self._compute_fault_correlation_attention(weighted_C, self.r)

        # 融合函数级 + 故障关联权重（两者协同放大关键函数）
        combined_func_weights = self.func_attention_weights * self.fault_corr_weights
        combined_func_weights = combined_func_weights / np.max(combined_func_weights)  # 归一化避免权重溢出
        weighted_C = weighted_C * combined_func_weights  # (n, m)
        self.C = weighted_C

        # 4. 筛选隐藏层中心（基于加权后的覆盖矩阵）
        self.hidden_centers = self._select_hidden_centers()
        h = len(self.hidden_centers)
        if h == 0:
            raise ValueError("未筛选出隐藏层中心，需调整β阈值")

        # 5. 计算全局sigma
        self.sigma = self._compute_global_sigma(self.hidden_centers)
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)

        # 6. 隐藏层注意力：抑制仅匹配成功用例的中心
        self.hidden_attention_weights = self._compute_hidden_attention(
            self.hidden_centers, self.C, self.r
        )

        # 7. 计算激活矩阵A（融入隐藏层注意力）
        A = []
        for c in self.C:
            activations = [np.exp(-(self._compute_wbc_distance(c, mu, count_l) ** 2) / (2 * self.sigma ** 2))
                           for mu in self.hidden_centers]
            weighted_activations = np.array(activations) * self.hidden_attention_weights
            A.append(weighted_activations)
        self.A = np.array(A)

        # 8. 求解输出层权重
        self.weights = pinv(self.A) @ self.r

    def compute_suspiciousness(self):
        if self.weights is None:
            raise ValueError("RBFNN未训练，无法计算可疑度")
        h = len(self.hidden_centers)
        count_l = np.sum(self.C, axis=0)
        count_l = np.where(count_l == 0, 1, count_l)
        suspiciousness = []
        for j in range(self.m):
            # 虚拟覆盖向量（仅激活第j个函数）+ 融合权重加权
            virtual_c = np.zeros(self.m)
            virtual_c[j] = 1
            # 应用函数级 + 故障关联融合权重
            virtual_c_weighted = virtual_c * self.func_attention_weights[j] * self.fault_corr_weights[j]

            # 计算虚拟激活值（融入隐藏层注意力）
            virtual_activations = [
                np.exp(-(self._compute_wbc_distance(virtual_c_weighted, mu, count_l) ** 2) / (2 * self.sigma ** 2))
                for mu in self.hidden_centers]
            virtual_activations = np.array(virtual_activations) * self.hidden_attention_weights
            virtual_activations = virtual_activations.reshape(1, -1)
            # print("虚拟激活值：", virtual_activations)
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

    # 构建覆盖矩阵并训练
    cov_matrix = torch.stack(matrix)
    shuffle_indices = torch.randperm(cov_matrix.size(0))
    shuffled_matrix = cov_matrix[shuffle_indices]
    count = (cov_matrix[:, -1] == 1).sum().item()
    print(f"覆盖矩阵形状: {cov_matrix.shape}")
    print(f"失败测试用例数: {count}")
    print(f"通过测试用例数: {len(cov_matrix) - count}")

    if cov_matrix is not None and cov_matrix.shape[0] > 1:
        # 初始化带四重注意力的RBFNN（优化故障关联权重缩放强度）
        rbfnn = RBFNNWithAttention(
            beta=0.035,
            func_attention_scale=2.0,  # 增强稀疏函数权重
            hidden_attention_scale=1.0,  # 增强隐藏层筛选
            noise_suppress_scale=2.0,  # 增强噪声用例抑制
            fault_corr_scale=1.0  # 放大仅失败覆盖函数的权重
        )
        rbfnn.train(cov_matrix)
        print("开始计算可疑度...")
        function_suspiciousness = rbfnn.compute_suspiciousness()
        print("可疑度计算完成")
        ranked_functions = rbfnn.rank_functions()

        # 计算文件平均可疑度
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

        # 输出到CSV
        with open(resultFile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'File', 'Score'])
            for rank, (filename, avg_susp) in enumerate(file_avg_susp, 1):
                writer.writerow([rank, filename, round(avg_susp, 6)])

        print(f"文件可疑度排序已输出到：{resultFile}")

    else:
        print("错误: 数据不足，无法训练模型")