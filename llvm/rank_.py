import math
import os
from collections import defaultdict
from configparser import ConfigParser
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

num = 5000  # top stmt num


def deleteGcdaPath(gcdaDirName):
    return re.sub(r'/CMakeFiles/[^/]+.dir', '', gcdaDirName)


class SuspiciousnessBFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout_rate=0.3):
        """
        BFNN - 基础前馈神经网络
        Args:
            input_size: 输入特征维度
            hidden_sizes: 隐藏层大小列表
            dropout_rate: dropout比率
        """
        super(SuspiciousnessBFNN, self).__init__()

        # 构建动态网络层
        layers = []
        prev_size = input_size

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.BatchNorm1d(hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # 输出层
        layers.append(nn.Linear(prev_size, input_size))

        self.network = nn.Sequential(*layers)
        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化参数"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.1)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, input_size]
        Returns:
            suspiciousness: 可疑度预测 [batch_size, input_size]
        """
        output = self.network(x)
        suspiciousness = torch.sigmoid(output)
        return suspiciousness


class ImprovedSuspiciousnessBFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 128, 64], dropout_rate=0.4):
        """
        改进的BFNN，更适合缺陷定位任务
        """
        super(ImprovedSuspiciousnessBFNN, self).__init__()

        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.BatchNorm1d(hidden_sizes[2]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(hidden_sizes[2], hidden_sizes[1]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[1], hidden_sizes[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_sizes[0], input_size),
        )

        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化参数"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0.1)

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)
        # 解码
        output = self.decoder(encoded)
        suspiciousness = torch.sigmoid(output)
        return suspiciousness


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
    print("开始训练BFNN模型...")

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

        def __len__(self):
            return len(self.features)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    dataset = CoverageDataset(features, cov_matrix[:, -1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(32, len(dataset)),
        shuffle=True
    )

    # 根据数据规模选择合适的模型
    if input_size <= 1000:
        # 小规模数据使用简单BFNN
        model = SuspiciousnessBFNN(
            input_size=input_size,
            hidden_sizes=[256, 128, 64],
            dropout_rate=0.3
        )
    else:
        # 大规模数据使用改进BFNN
        model = ImprovedSuspiciousnessBFNN(
            input_size=input_size,
            hidden_sizes=[512, 256, 128],
            dropout_rate=0.4
        )

    # 优化器 - 使用更大的学习率和权重衰减
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01  # 更强的正则化
    )

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=15,
        factor=0.5,
        min_lr=1e-6
    )

    # 训练过程
    model.train()
    best_loss = float('inf')
    patience = 25
    patience_counter = 0

    for epoch in range(300):
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
                    # 失败用例：被覆盖的语句应该预测为高可疑度(接近1)
                    loss += torch.mean((1 - predictions[i][covered]) ** 2)
                else:  # 通过测试用例
                    covered = (batch_features[i] == 1)
                    # 通过用例：被覆盖的语句应该预测为低可疑度(接近0)
                    loss += torch.mean(predictions[i][covered] ** 2)

            loss = loss / batch_size
            loss.backward()

            # 梯度裁剪
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
            print(f'Epoch [{epoch + 1}/300], Loss: {avg_epoch_loss:.4f}, LR: {current_lr:.6f}')

    # 只计算文件级别的可疑度
    ranked_files, file_scores = get_file_suspiciousness(model, cov_matrix, stmtmap)

    print("\n训练完成!")

    # 保存文件级别结果
    save_file_results(ranked_files, bugId)

    return model, ranked_files


def save_file_results(ranked_files, bug_id):
    """保存文件级别可疑度结果"""
    file_output_file = f"/home/chris/FLL-workplace/llvmbugs/result/{bug_id}/NNresult_file_{bug_id}.txt"
    with open(file_output_file, 'w', encoding='utf-8') as f:
        f.write(f"文件级别缺陷定位结果 - Bug {bug_id}\n")
        f.write("=" * 130 + "\n")
        f.write(f"{'排名':<6} {'平均可疑度':<12} {'语句数':<8} {'高可疑':<6} {'中可疑':<6} {'低可疑':<6} {'文件名'}\n")
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
