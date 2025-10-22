import math
import os
from collections import defaultdict
from configparser import ConfigParser
import re
import torch
from sklearn.cluster import KMeans

num = 5000  # top stmt num


def deleteGcdaPath(gcdaDirName):
    return re.sub(r'/CMakeFiles/[^/]+.dir', '', gcdaDirName)


import torch.nn as nn
import torch.nn.functional as F


class SuspiciousnessRBFNN(nn.Module):
    def __init__(self, input_size, num_centers=50, sigma=1.0):
        super(SuspiciousnessRBFNN, self).__init__()
        self.input_size = input_size
        self.num_centers = num_centers
        self.sigma = nn.Parameter(torch.tensor(sigma))

        # RBF层参数
        self.centers = nn.Parameter(torch.randn(num_centers, input_size))
        self.linear = nn.Linear(num_centers, input_size)  # 输出层

        # 初始化参数
        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.centers)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.1)

    def rbf_function(self, x, centers):
        """径向基函数 - 高斯核"""
        # 计算输入与中心点的欧氏距离
        x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        centers = centers.unsqueeze(0)  # [1, num_centers, input_size]

        distances = torch.sum((x - centers) ** 2, dim=2)  # [batch_size, num_centers]

        # 应用高斯径向基函数
        rbf_output = torch.exp(-distances / (2 * self.sigma ** 2))
        return rbf_output

    def forward(self, x):
        # RBF层
        rbf_output = self.rbf_function(x, self.centers)  # [batch_size, num_centers]

        # 线性输出层
        output = self.linear(rbf_output)  # [batch_size, input_size]

        # 应用sigmoid确保输出在[0,1]范围内
        suspiciousness = torch.sigmoid(output)
        return suspiciousness


class ImprovedSuspiciousnessRBFNN(nn.Module):
    def __init__(self, input_size, num_centers=100, hidden_size=64):
        super(ImprovedSuspiciousnessRBFNN, self).__init__()
        self.input_size = input_size
        self.num_centers = num_centers

        # RBF层
        self.centers = nn.Parameter(torch.randn(num_centers, input_size))
        self.sigma = nn.Parameter(torch.ones(num_centers))

        # 隐藏层
        self.hidden1 = nn.Linear(num_centers, hidden_size)
        self.hidden2 = nn.Linear(hidden_size, hidden_size // 2)

        # 输出层
        self.output_layer = nn.Linear(hidden_size // 2, input_size)

        # 激活函数和正则化
        self.dropout = nn.Dropout(0.2)
        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.centers)
        nn.init.xavier_uniform_(self.hidden1.weight)
        nn.init.xavier_uniform_(self.hidden2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.constant_(self.sigma, 1.0)

    def rbf_function(self, x):
        """改进的径向基函数"""
        x = x.unsqueeze(1)  # [batch_size, 1, input_size]
        centers = self.centers.unsqueeze(0)  # [1, num_centers, input_size]

        # 计算距离
        distances = torch.sum((x - centers) ** 2, dim=2)  # [batch_size, num_centers]

        # 每个中心点使用自己的sigma参数
        sigma_matrix = self.sigma.unsqueeze(0)  # [1, num_centers]
        rbf_output = torch.exp(-distances / (2 * sigma_matrix ** 2))

        return rbf_output

    def forward(self, x):
        # RBF层
        rbf_output = self.rbf_function(x)

        # 隐藏层
        x = F.relu(self.hidden1(rbf_output))
        x = self.dropout(x)
        x = F.relu(self.hidden2(x))
        x = self.dropout(x)

        # 输出层
        output = self.output_layer(x)
        suspiciousness = torch.sigmoid(output)

        return suspiciousness


def initialize_rbf_centers_kmeans(features, num_centers):
    """使用K-means初始化RBF中心点"""
    features_np = features.cpu().numpy()
    kmeans = KMeans(n_clusters=num_centers, random_state=42, n_init=10)
    kmeans.fit(features_np)
    centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
    return centers


def get_file_suspiciousness(model, cov_matrix, stmtmap):
    """直接计算文件级别的可疑度"""
    model.eval()
    with torch.no_grad():
        features = cov_matrix[:, :-1]
        all_predictions = model(features)
        avg_suspiciousness = torch.mean(all_predictions, dim=0)

        # 按文件分组语句可疑度
        file_scores = defaultdict(list)

        # 创建反向映射：索引 -> (文件名, 语句)
        index_to_statement = {idx: key for key, idx in stmtmap.items()}

        # 收集每个文件的语句可疑度
        for idx in range(avg_suspiciousness.shape[0]):
            if idx in index_to_statement:
                statement_key = index_to_statement[idx]
                filename, statement = statement_key.split(',', 1)
                score = avg_suspiciousness[idx].item()
                file_scores[filename].append(score)

        # 计算每个文件的统计信息
        file_stats = {}
        for filename, scores in file_scores.items():
            if scores:  # 确保有语句得分
                file_stats[filename] = {
                    'avg_score': sum(scores) / len(scores),
                    'stmt_count': len(scores),
                    'max_score': max(scores),
                    'min_score': min(scores),
                    'high_suspicion_count': len([s for s in scores if s > 0.7]),
                    'medium_suspicion_count': len([s for s in scores if 0.3 <= s <= 0.7]),
                    'low_suspicion_count': len([s for s in scores if s < 0.3])
                }

        # 按平均可疑度排序
        ranked_files = sorted(file_stats.items(),
                              key=lambda x: x[1]['avg_score'],
                              reverse=True)

        return ranked_files, file_stats


def complete_training_pipeline(cov_matrix, stmtmap, bugId):
    print("开始训练RBFNN模型...")
    # 准备数据
    features = cov_matrix[:, :-1]
    input_size = features.shape[1]

    print(f"输入特征维度: {input_size}")
    print(f"训练样本数: {cov_matrix.shape[0]}")

    # 创建数据集
    class CoverageDataset(torch.utils.data.Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self): return len(self.features)

        def __getitem__(self, idx): return self.features[idx], self.labels[idx]

    dataset = CoverageDataset(features, cov_matrix[:, -1])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=min(32, len(dataset)), shuffle=True)

    # 确定RBF中心点数量
    num_centers = min(100, max(20, features.shape[0] // 10))
    print(f"使用 {num_centers} 个RBF中心点")

    # 初始化模型
    model = ImprovedSuspiciousnessRBFNN(
        input_size=input_size,
        num_centers=num_centers,
        hidden_size=64
    )

    # 使用K-means初始化中心点（可选）
    if features.shape[0] > num_centers:
        with torch.no_grad():
            initialized_centers = initialize_rbf_centers_kmeans(features, num_centers)
            model.centers.data = initialized_centers

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # 训练过程
    model.train()
    best_loss = float('inf')
    patience = 20
    patience_counter = 0

    for epoch in range(500):
        epoch_loss = 0
        model.train()

        for batch_features, batch_labels in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_features)

            loss = 0
            batch_size = batch_labels.shape[0]
            for i in range(batch_size):
                if batch_labels[i] == 1:  # 失败测试用例
                    covered = (batch_features[i] == 1)
                    if covered.sum() > 0:
                        loss += torch.mean((1 - predictions[i][covered]) ** 2)
                else:  # 通过测试用例
                    covered = (batch_features[i] == 1)
                    if covered.sum() > 0:
                        loss += torch.mean(predictions[i][covered] ** 2)

            # 添加正则化项
            reg_loss = 0.001 * torch.norm(model.centers)
            loss = loss / batch_size + reg_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_epoch_loss)

        # 早停机制
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"早停在epoch {epoch}, 最佳损失: {best_loss:.4f}")
            model.load_state_dict(best_model_state)
            break

        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch + 1}/500], Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}')

    # 只计算文件级别的可疑度
    ranked_files, file_scores = get_file_suspiciousness(model, cov_matrix, stmtmap)

    print("\n训练完成!")

    # 保存文件级别结果
    save_file_results(ranked_files, bugId)

    return model, ranked_files


def save_file_results(ranked_files, bug_id):
    """保存文件级别可疑度结果"""
    file_output_file = f"file_suspiciousness_bug_{bug_id}.txt"
    with open(file_output_file, 'w', encoding='utf-8') as f:
        f.write(f"文件级别缺陷定位结果 - Bug {bug_id}\n")
        f.write("=" * 130 + "\n")
        f.write(
            f"{'排名':<6} {'平均可疑度':<12} {'语句数':<8} {'高可疑':<6} {'中可疑':<6} {'低可疑':<6} {'文件名'}\n")
        f.write("=" * 130 + "\n")
        for i, (filename, score_info) in enumerate(ranked_files):
            f.write(f"{i + 1:<6} {score_info['avg_score']:<12.6f} {score_info['stmt_count']:<8} "
                    f"{score_info['high_suspicion_count']:<6} {score_info['medium_suspicion_count']:<6} "
                    f"{score_info['low_suspicion_count']:<6} {filename}\n")

    print(f"文件级别结果已保存到: {file_output_file}")


def fileRank(bugId, rev, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    baseInfoDir = cfg.get('llvm-locations', 'infodir')
    basePassDir = cfg.get('llvm-locations', 'passdir')
    baseResultDir = cfg.get('llvm-locations', 'resultFile')
    resultFile = baseResultDir + bugId + '/resultFile_file.csv'
    if not os.path.exists(baseResultDir + bugId):
        os.mkdir(baseResultDir + bugId)

    # SBFL
    failfile = open(baseInfoDir + bugId + '/failcov/stmt_info.txt')
    faillines = failfile.readlines()
    failfile.close()
    # 初始化数据结构
    failstmt = dict()  # 记录每个语句被失败用例覆盖的次数
    passstmt = dict()  # 记录每个语句被通过用例覆盖的次数
    failfileset = set()  # 失败用例覆盖的文件集合
    failfilemapstmt = dict()  # 每个文件对应的语句集合映射
    # 处理失败用例的覆盖率数据
    for i in range(len(faillines)):
        faillinesplit = faillines[i].strip().split(',')
        filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', faillinesplit[0]).split('.gcda')[0]
        if not filename.endswith('.cpp'):  # 只处理.cpp文件
            continue
        failfileset.add(filename)  # 添加到失败文件集合
        stmtlist = faillines[i].strip().split(':')[1].split(',')  # 提取语句列表
        failfilemapstmt[filename] = set(stmtlist)  # 建立文件到语句的映射
        # 初始化每个语句的计数
        for stmt in stmtlist:
            failstmt[f"{filename},{stmt}"] = 1  # 失败用例覆盖1次
            passstmt[f"{filename},{stmt}"] = 0  # 通过用例覆盖0次（初始）
    # 处理所有通过测试用例的覆盖率
    for i in os.listdir(basePassDir + '/' + bugId + '/passcov'):
        # 读取单个通过用例的覆盖率文件
        passfile = open(basePassDir + '/' + bugId + '/passcov/' + i + '/stmt_info.txt')
        passlines = passfile.readlines()
        passfile.close()

        # 处理通过用例的覆盖率数据
        for j in range(len(passlines)):
            passlinesplit = passlines[j].strip().split(',')
            filename = passlinesplit[0].strip().split('.gcda')[0].strip()
            filename = deleteGcdaPath(filename)
            if not filename.endswith('.cpp'):  # 只考虑.cpp文件
                continue
            if filename not in failfileset:  # 只处理失败用例也覆盖的文件
                continue
            stmtlist = passlines[j].strip().split(':')[1].split(',')
            # 统计通过用例覆盖的语句（只统计失败用例也覆盖的语句）
            for stmt in set(stmtlist) & failfilemapstmt[filename]:
                passstmt[f"{filename},{stmt}"] += 1  # 增加通过用例计数
    # 计算每个语句的可疑度得分
    score = dict()
    for key in failstmt.keys():
        score[key] = float(failstmt[key]) / math.sqrt(float(failstmt[key]) * (failstmt[key] + passstmt[key]))
    score = sorted(score.items(), key=lambda x: x[1], reverse=True)[:num]
    score_set = {stmt for stmt, _ in score}

    # RBFNN
    # original test program fail.c
    stmtmap = dict()
    matrix = []
    with open(baseInfoDir + bugId + '/failcov/stmt_info.txt', 'r') as f:
        lines = f.readlines()
        vector = torch.zeros(num + 1)
        vector[-1] = 1
        for i in range(len(lines)):
            items = lines[i].split(':')
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            stmtlist = items[1].split(',')
            for j in range(len(stmtlist)):
                if f"{filename},{stmtlist[j]}" in score_set:
                    vector[len(stmtmap)] = 1
                    stmtmap[f"{filename},{stmtlist[j]}"] = len(stmtmap)
        matrix.append(vector)

    failcovDir = basePassDir + bugId + '/failcov/'

    # fail test program
    if os.path.exists(failcovDir):
        for dir in os.listdir(failcovDir):
            vector = torch.zeros(num + 1)
            vector[-1] = 1
            stmt_info = open(failcovDir + dir + '/stmt_info.txt')
            stmtlines = stmt_info.readlines()
            stmt_info.close()
            for i in range(len(stmtlines)):
                items = stmtlines[i].split(':')
                filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                stmtlist = items[1].split(',')
                for j in range(len(stmtlist)):
                    if f"{filename},{stmtlist[j]}" in stmtmap:
                        vector[stmtmap[f"{filename},{stmtlist[j]}"]] = 1
            matrix.append(vector)

    # success test program
    passcovDir = basePassDir + bugId + '/passcov/'
    for dir in os.listdir(passcovDir):
        vector = torch.zeros(num + 1)
        stmt_info = open(passcovDir + dir + '/stmt_info.txt')
        stmtlines = stmt_info.readlines()
        stmt_info.close()
        for i in range(len(stmtlines)):
            items = stmtlines[i].split(':')
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            stmtlist = items[1].split(',')
            for j in range(len(stmtlist)):
                if f"{filename},{stmtlist[j]}" in stmtmap:
                    vector[stmtmap[f"{filename},{stmtlist[j]}"]] = 1
        matrix.append(vector)
    cov_matrix = torch.stack(matrix)
    indices = torch.randperm(cov_matrix.shape[0])
    shuffled_matrix = cov_matrix[indices]

    count = (shuffled_matrix[:, -1] == 1).sum().item()
    print(shuffled_matrix)
    print(f"覆盖矩阵形状: {shuffled_matrix.shape}")
    print(f"失败测试用例数: {count}")
    print(f"通过测试用例数: {len(shuffled_matrix) - count}")

    if shuffled_matrix is not None and shuffled_matrix.shape[0] > 1:
        model, ranked_files = complete_training_pipeline(shuffled_matrix, stmtmap, bugId)
        print(f"\n=== 缺陷定位结果摘要 ===")
        print(f"分析的文件数量: {len(ranked_files)}")
        print(f"最可疑的文件: {ranked_files[0][0] if ranked_files else '无'}")
        print(f"最高平均可疑度: {ranked_files[0][1]['avg_score']:.6f if ranked_files else 0}")

    else:
        print("错误: 数据不足，无法训练模型")
