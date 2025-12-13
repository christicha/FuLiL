import os
import csv
from configparser import ConfigParser
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class GatedBipartiteGNNLayer(nn.Module):
    """
    引入门控机制的二部图 GNN 消息传递层 (类似 GRU 的更新机制)。
    """

    def __init__(self, in_ft_E, in_ft_T, out_ft):
        super(GatedBipartiteGNNLayer, self).__init__()

        self.W_E = nn.Linear(in_ft_E, out_ft, bias=True)
        self.W_T = nn.Linear(in_ft_T, out_ft, bias=False)

        # 门控机制：用于计算门控值
        self.gate_linear = nn.Linear(2 * out_ft, out_ft)

    def forward(self, H_E, H_T, C_norm_T2E):
        # 1. T -> E 消息传递
        # T 节点特征 H_T (N_T x in_ft_T) 通过 C_norm_T2E 聚合到 E 节点
        message_T = C_norm_T2E @ H_T  # (N_E x in_ft_T)

        # 2. 消息和旧状态的线性变换
        H_E_transformed = self.W_E(H_E)  # (N_E x out_ft)
        M_T_transformed = self.W_T(message_T)  # (N_E x out_ft)

        # 3. 候选状态
        H_E_candidate = torch.tanh(H_E_transformed + M_T_transformed)  # (N_E x out_ft)

        # 4. 门控机制
        combined = torch.cat([H_E_transformed, M_T_transformed], dim=1)
        gate = torch.sigmoid(self.gate_linear(combined))  # (N_E x out_ft)

        # 5. 更新状态 (GRU-like Update)
        H_E_new = gate * H_E_candidate + (1 - gate) * H_E_transformed

        return H_E_new


class GNN_FL(nn.Module):
    def __init__(self, sbfl_features_size, hidden_size=64):
        super(GNN_FL, self).__init__()

        self.T_feat_dim = 1  # 测试结果 R (0 或 1)
        self.hidden_size = hidden_size

        # E-node 初始特征投影
        self.proj_E = nn.Linear(sbfl_features_size, hidden_size)

        # 3 层门控 GNN
        self.gnn1 = GatedBipartiteGNNLayer(hidden_size, self.T_feat_dim, hidden_size)
        self.gnn2 = GatedBipartiteGNNLayer(hidden_size, self.T_feat_dim, hidden_size)
        self.gnn3 = GatedBipartiteGNNLayer(hidden_size, self.T_feat_dim, hidden_size)  # 新增 GNN 层

        # 每层 GNN 后的层归一化 (Layer Normalization)
        # LayerNorm 针对 (N_E x hidden_size) 的特征进行归一化
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)  # 新增 LayerNorm 层

        # 最终实体得分层
        self.entity_score_linear = nn.Linear(hidden_size, 1)

    def _get_C_norm(self, C):
        """计算归一化覆盖矩阵 C_norm_T2E (列归一化)"""
        D_E = torch.sum(C.float(), dim=0, keepdim=True)
        D_E_inv = torch.where(D_E == 0, torch.tensor(1.0, device=C.device), 1.0 / D_E)

        C_norm = C.float() * D_E_inv
        C_norm_T2E = C_norm.t()
        return C_norm_T2E

    def forward(self, C, R, H_E0):
        H_T = R.float()

        # 初始投影
        H_E = F.relu(self.proj_E(H_E0))

        C_norm_T2E = self._get_C_norm(C)

        # GNN Layers with Layer Normalization and ReLU

        # 1st Layer
        H_E = self.gnn1(H_E, H_T, C_norm_T2E)
        H_E = F.relu(self.ln1(H_E))

        # 2nd Layer
        H_E = self.gnn2(H_E, H_T, C_norm_T2E)
        H_E = F.relu(self.ln2(H_E))

        # 3rd Layer (新增)
        H_E = self.gnn3(H_E, H_T, C_norm_T2E)
        H_E = F.relu(self.ln3(H_E))

        # 输出原始 Logits (N_E x 1)
        entity_scores_raw = self.entity_score_linear(H_E)

        suspiciousness_scores = torch.sigmoid(entity_scores_raw).squeeze(1)

        return suspiciousness_scores, entity_scores_raw.squeeze(1)

    def predict_suspiciousness(self, C, R, H_E0):
        """仅用于推理阶段，返回最终可疑度分数"""
        with torch.no_grad():
            _, entity_scores_raw_logits = self.forward(C, R, H_E0)
            return torch.sigmoid(entity_scores_raw_logits)

def fileRank_GNN(bugId, rev, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')

    if not os.path.exists(os.path.join(baseResultDir, bugId)):
        os.makedirs(os.path.join(baseResultDir, bugId))
    resultFile = os.path.join(baseResultDir, bugId, 'result_gnn.csv')

    matrix = []
    methodmap = dict()

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

    # --- SBFL 特征计算 ---
    R_fail_mask = (R_data.squeeze(1) == 1)
    R_pass_mask = (R_data.squeeze(1) == 0)

    C_fail = C_data[R_fail_mask]
    C_pass = C_data[R_pass_mask]

    a_ef = torch.sum(C_fail, dim=0).float()
    a_ep = torch.sum(C_pass, dim=0).float()
    a_nf = C_fail.shape[0] - a_ef
    a_np = C_pass.shape[0] - a_ep

    H_E0_raw = torch.stack([a_ef, a_ep, a_nf, a_np], dim=1)  # (N_E x 4)

    H_E0_max = H_E0_raw.max(dim=0, keepdim=True).values
    H_E0 = H_E0_raw / (H_E0_max + 1e-6)
    H_E0 = H_E0.to(device)
    SBFL_FEATURES_SIZE = H_E0.shape[1]

    # --- 标签定义 ---
    Y_E_fault_mask = (a_ef > 0) & (a_ep == 0)

    if Y_E_fault_mask.sum().item() == 0:
        Y_E_fault_mask = (a_ef > 0)

    idx_faulty = torch.where(Y_E_fault_mask.to(device))[0]
    idx_clean = torch.where(~Y_E_fault_mask.to(device))[0]

    # --- 模型初始化 ---
    model = GNN_FL(SBFL_FEATURES_SIZE).to(device)

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EPOCHS = 10
    MARGIN = 0.5  # 排名损失的边距
    print("Starting GNN Training")

    model.train()
    C_data, R_data = C_data.to(device), R_data.to(device)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()

        # 得到原始 Logits
        _, entity_scores_raw_logits = model(C_data, R_data, H_E0)

        # --- 计算 Margin Ranking Loss ---
        if len(idx_faulty) > 0 and len(idx_clean) > 0:
            S_positive = entity_scores_raw_logits[idx_faulty].unsqueeze(1)  # (N_pos x 1)
            S_negative = entity_scores_raw_logits[idx_clean].unsqueeze(0)  # (1 x N_neg)

            # Ranking Loss: L = mean( max(0, margin - (S_pos - S_neg)) )
            difference = S_positive - S_negative
            ranking_loss = torch.relu(MARGIN - difference).mean()
        else:
            ranking_loss = torch.sum(entity_scores_raw_logits ** 2) * 1e-4  # L2正则化作为占位符

        loss = ranking_loss

        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Ranking Loss: {loss.item():.6f}")

    # --- 故障定位 (推理) ---
    print("Calculating Suspiciousness from 3-Layer GNN Entity Embeddings...")
    model.eval()
    function_suspiciousness_tensor = model.predict_suspiciousness(C_data, R_data, H_E0)
    function_suspiciousness = function_suspiciousness_tensor.cpu().numpy()

    # --- 结果聚合与输出 ---
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

        print(f"GNN-FL (3-Layer GGNN + LayerNorm, Rank) 文件可疑度排序已输出到：{resultFile}")

    except Exception as e:
        print(f"Error saving CSV: {e}")