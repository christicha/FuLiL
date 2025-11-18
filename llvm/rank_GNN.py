import os
import csv
from configparser import ConfigParser
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


# ==========================================
# 1. GNN 模型定义 (基于 Bipartite Graph)
# ==========================================

class BipartiteGNNLayer(nn.Module):
    """
    简化的二部图 GNN 消息传递层：T-Node (测试) -> E-Node (实体/方法) 聚合。
    """

    def __init__(self, in_ft_E, in_ft_T, out_ft):
        super(BipartiteGNNLayer, self).__init__()
        self.W_E = nn.Linear(in_ft_E, out_ft, bias=False)
        self.W_T = nn.Linear(in_ft_T, out_ft, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_ft))

    def forward(self, H_E, H_T, C_norm_T2E):
        # H_E: 实体/方法节点特征 (N_E x in_ft_E)
        # H_T: 测试用例节点特征 (N_T x in_ft_T)
        # C_norm_T2E: 归一化覆盖矩阵的转置 (N_E x N_T)

        # T -> E 消息传递 (N_E x N_T) @ (N_T x in_ft_T) = (N_E x in_ft_T)
        message_T = C_norm_T2E @ H_T

        transformed_message_T = self.W_T(message_T)
        transformed_H_E = self.W_E(H_E)

        H_E_new = F.relu(transformed_H_E + transformed_message_T + self.bias)
        return H_E_new


class GNN_FL(nn.Module):
    def __init__(self, sbfl_features_size, hidden_size=64):
        super(GNN_FL, self).__init__()

        self.hidden_size = hidden_size
        self.T_feat_dim = 1  # 测试结果 R (0 或 1)

        self.proj_E = nn.Linear(sbfl_features_size, hidden_size)
        self.gnn1 = BipartiteGNNLayer(hidden_size, self.T_feat_dim, hidden_size)
        self.gnn2 = BipartiteGNNLayer(hidden_size, self.T_feat_dim, hidden_size)

        # 修正：最终实体得分层不使用 Sigmoid，输出 Logits
        self.entity_score_linear = nn.Linear(hidden_size, 1)

    def _get_C_norm(self, C):
        """计算用于消息传递的归一化覆盖矩阵 C_norm_T2E (列归一化)"""
        D_E = torch.sum(C.float(), dim=0, keepdim=True)  # (1 x N_E)
        D_E_inv = torch.where(D_E == 0, torch.tensor(1.0, device=C.device), 1.0 / D_E)  # (1 x N_E)

        C_norm = C.float() * D_E_inv
        C_norm_T2E = C_norm.t()
        return C_norm_T2E

    def forward(self, C, R, H_E0):
        # C: 覆盖矩阵 (N_T x N_E), R: 测试结果 (N_T x 1)
        H_T = R.float()
        H_E = F.relu(self.proj_E(H_E0))

        C_norm_T2E = self._get_C_norm(C)

        H_E = self.gnn1(H_E, H_T, C_norm_T2E)
        H_E = self.gnn2(H_E, H_T, C_norm_T2E)

        # 1. 可疑度预测：输出 Logits
        entity_scores_raw = self.entity_score_linear(H_E)  # (N_E x 1)

        # 2. Readout: 预测测试用例结果：输出 Logits (用于 BCEWithLogitsLoss)
        predicted_R_raw = C.float() @ entity_scores_raw  # (N_T x 1)

        # 在推理时，对可疑度 Logits 应用 Sigmoid 得到 [0, 1] 范围分数
        suspiciousness_scores = torch.sigmoid(entity_scores_raw).squeeze(1)

        # 返回可疑度 (Sigmoid 后的分数) 和 预测的 R (Logits)
        return suspiciousness_scores, predicted_R_raw.squeeze(1)

    def predict_suspiciousness(self, C, R, H_E0):
        """仅用于推理阶段，返回最终可疑度分数"""
        with torch.no_grad():
            H_T = R.float()
            H_E = F.relu(self.proj_E(H_E0))

            C_norm_T2E = self._get_C_norm(C)

            H_E = self.gnn1(H_E, H_T, C_norm_T2E)
            H_E = self.gnn2(H_E, H_T, C_norm_T2E)

            entity_scores_raw = self.entity_score_linear(H_E)
            suspiciousness_scores = torch.sigmoid(entity_scores_raw).squeeze(1)
            return suspiciousness_scores


# ==========================================
# 2. 主逻辑函数
# ==========================================
def fileRank_GNN(bugId, rev, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')

    if not os.path.exists(os.path.join(baseResultDir, bugId)):
        os.makedirs(os.path.join(baseResultDir, bugId))
    resultFile = os.path.join(baseResultDir, bugId, 'GNNresultFile_file.csv')

    matrix = []
    methodmap = dict()

    # -----------------------------
    # 数据读取部分
    # -----------------------------
    # (此处省略读取代码，假设读取结果为 full_data, C_data, R_data, methodmap)
    # ... (您的数据读取逻辑不变)

    # -----------------------------
    # 简化的数据读取逻辑占位符
    # -----------------------------
    try:
        # 读取失败测试用例信息
        fail_info_path = baseInfoDir + bugId + '/failcov/method_info.txt'
        if os.path.exists(fail_info_path):
            current_methods = set()
            with open(fail_info_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    items = line.strip().split(',')
                    if len(items) < 2: continue
                    filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                    methodname = items[1]
                    key = f"{filename},{methodname}"
                    current_methods.add(key)
                    if key not in methodmap:
                        methodmap[key] = len(methodmap)

                # 重新构建失败用例向量 (确保所有方法都被映射)
                vector = torch.zeros(len(methodmap) + 1)
                vector[-1] = 1  # 失败标签
                for key, index in methodmap.items():
                    if key in current_methods:
                        vector[index] = 1
                matrix.append(vector)

        # 读取成功测试用例
        cnt = 0
        passcovDir = basePassDir + bugId + '/passcov/'
        if os.path.exists(passcovDir):
            for dir_name in os.listdir(passcovDir):
                vector = torch.zeros(len(methodmap) + 1)
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

        # 读取剩余失败测试用例
        failcovDir = basePassDir + bugId + '/failcov'
        i = 0
        if os.path.exists(failcovDir):
            for dir_name in os.listdir(failcovDir):
                if i >= cnt - 1: break
                vector = torch.zeros(len(methodmap) + 1)
                vector[-1] = 1
                method_file = os.path.join(failcovDir, dir_name, 'method_info.txt')
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
                i += 1

    except Exception as e:
        print(f"数据读取失败: {e}")
        return

    if not matrix:
        print("Error: Matrix is empty.")
        return

    full_data = torch.stack(matrix)

    C_data = full_data[:, :-1]
    R_data = full_data[:, -1].unsqueeze(1)

    N_T, N_E = C_data.shape
    if N_T == 0 or N_E == 0:
        print("错误: 覆盖矩阵维度为零。")
        return

    print(f"Data loaded: {N_T} test cases, {N_E} methods/entities.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------
    # 1. 构造 E-node 初始特征 H_E0 (SBFL 统计量)
    # ---------------------------------------------
    R_fail_mask = (R_data.squeeze(1) == 1)
    R_pass_mask = (R_data.squeeze(1) == 0)

    C_fail = C_data[R_fail_mask]
    C_pass = C_data[R_pass_mask]

    a_ef = torch.sum(C_fail, dim=0).float()
    a_ep = torch.sum(C_pass, dim=0).float()
    a_nf = C_fail.shape[0] - a_ef
    a_np = C_pass.shape[0] - a_ep

    H_E0_raw = torch.stack([a_ef, a_ep, a_nf, a_np], dim=1)  # (N_E x 4)

    # 修正：对初始特征进行归一化
    H_E0_max = H_E0_raw.max(dim=0, keepdim=True).values
    H_E0 = H_E0_raw / (H_E0_max + 1e-6)  # 归一化到 [0, 1] 范围
    H_E0 = H_E0.to(device)
    SBFL_FEATURES_SIZE = H_E0.shape[1]  # 4

    # ---------------------------------------------
    # 2. 初始化 GNN 模型与训练
    # ---------------------------------------------
    model = GNN_FL(SBFL_FEATURES_SIZE).to(device)

    # 修正：使用 BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()

    # 修正：使用更小的学习率
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 100  # 增加 Epochs 以确保收敛
    print("Starting GNN Training...")

    model.train()
    C_data, R_data = C_data.to(device), R_data.to(device)

    # 真实标签 R_data 扁平化为 (N_T)
    R_target = R_data.float().squeeze(1)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # predicted_R_raw 是 Logits (N_T)
        susp_scores, predicted_R_raw = model(C_data, R_data, H_E0)

        # 损失函数直接作用于 Logits 和目标 R_target
        loss = criterion(predicted_R_raw, R_target)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")

    # -----------------------------
    # 故障定位 (Suspiciousness Calculation)
    # -----------------------------

    print("Calculating Suspiciousness from GNN Entity Embeddings...")
    model.eval()
    function_suspiciousness_tensor = model.predict_suspiciousness(C_data, R_data, H_E0)
    function_suspiciousness = function_suspiciousness_tensor.cpu().numpy()

    # -----------------------------
    # 结果聚合与输出
    # -----------------------------

    index_to_method = {v: k for k, v in methodmap.items()}
    file_susp_dict = dict()

    for func_idx in range(len(function_suspiciousness)):
        method_key = index_to_method.get(func_idx)
        if not method_key: continue
        filename = method_key.split(',', 1)[0]
        susp = function_suspiciousness[func_idx]
        if filename not in file_susp_dict:
            file_susp_dict[filename] = []
        file_susp_dict[filename].append(susp)

    file_avg_susp = []
    for filename, susp_list in file_susp_dict.items():
        if not susp_list: continue
        avg_susp = np.mean(susp_list)
        file_avg_susp.append((filename, avg_susp))
    file_avg_susp.sort(key=lambda x: x[1], reverse=True)

    try:
        with open(resultFile, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Rank', 'File', 'Score'])
            for rank, (filename, avg_susp) in enumerate(file_avg_susp, 1):
                writer.writerow([rank, filename, round(avg_susp, 6)])

        print(f"GNN-FL 文件可疑度排序已输出到：{resultFile}")

    except Exception as e:
        print(f"Error saving CSV: {e}")