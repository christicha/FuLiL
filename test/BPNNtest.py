import os
import re
from collections import defaultdict
from itertools import islice
import torch
import numpy as np
from sklearn.cluster import KMeans

from Util import findFunByLine

bugIds = []
revisions = []
with open('testlist.txt', 'r') as f:
    for line in f:
        line = line.strip()
        items = line.split(',')
        bugIds.append(items[0])
        revisions.append(items[1])

baseResultDir = '/home/chris/FLL-workplace/llvmbugs/result/'
baseInfoDir = '/home/chris/FLL-workplace/llvmbugs/info/'
basePassDir = '/home/chris/FLL-workplace/llvmbugs/passdir/'
compileBasePath = '/home/chris/RecBi-workplace/llvmbugs/compilers/'

for i in range(len(bugIds)):
    bugId = bugIds[i]
    revision = revisions[i]
    print(f"bugid:{bugId} revision:{revision} \n")
    fileset = set()
    stmtmap = {}
    failfuncmapstmt = defaultdict(set)
    # original test program fail.c
    with open(baseResultDir + bugId + '/resultFile_file.csv', 'r') as f:
        for line in islice(f, 1, 21):
            line = line.strip()
            items = line.split(',')
            fileset.add(items[1])

    with open(baseInfoDir + bugId + '/failcov/stmt_info.txt', 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            items = lines[i].split(':')
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            compilePath = compileBasePath + revision + '/' + revision + '/llvm/'
            if filename in fileset:
                stmt = items[1].split(',')
                line_to_func = findFunByLine(compilePath + filename, stmt)
                for line, func in line_to_func.items():
                    failfuncmapstmt[func].add(f"{filename},{line}")
                for j in range(len(stmt)):
                    stmtmap[f"{filename},{stmt[j]}"] = len(stmtmap)

    num_statements = len(stmtmap) + 1
    fail_row = torch.ones(num_statements)
    matrix_rows = [fail_row]

    # fail test program
    failcovDir = basePassDir + bugId + '/failcov/'
    if os.path.exists(failcovDir):
        for test_dir in os.listdir(failcovDir):
            vector = torch.zeros(num_statements)
            vector[-1] = 1
            stmt_info = open(failcovDir + test_dir + '/stmt_info.txt')
            stmtlines = stmt_info.readlines()
            stmt_info.close()
            for i in range(len(stmtlines)):
                items = stmtlines[i].split(':')
                filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
                if filename in fileset:
                    stmt = items[1].split(',')
                    for j in range(len(stmt)):
                        if f"{filename},{stmt[j]}" in stmtmap:
                            vector[stmtmap[f"{filename},{stmt[j]}"]] = 1
            matrix_rows.append(vector)

    # success test program
    passcovDir = basePassDir + bugId + '/passcov/'
    for test_dir in os.listdir(passcovDir):
        vector = torch.zeros(num_statements)
        stmt_info = open(passcovDir + test_dir + '/stmt_info.txt')
        stmtlines = stmt_info.readlines()
        stmt_info.close()
        for i in range(len(stmtlines)):
            items = stmtlines[i].split(':')
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
            if filename in fileset:
                stmt = items[1].split(',')
                for j in range(len(stmt)):
                    if f"{filename},{stmt[j]}" in stmtmap:
                        vector[stmtmap[f"{filename},{stmt[j]}"]] = 1
        matrix_rows.append(vector)

    if matrix_rows:
        cov_matrix = torch.stack(matrix_rows)

    count = (cov_matrix[:, -1] == 1).sum().item()
    print(f"覆盖矩阵形状: {cov_matrix.shape}")
    print(f"失败测试用例数: {count}")
    print(f"通过测试用例数: {len(matrix_rows) - count}")

    import torch.nn as nn
    import torch.nn.functional as F


    class SuspiciousnessRBFNN(nn.Module):
        def __init__(self, input_size, num_centers=50, sigma=1.0):
            """
            RBF神经网络
            Args:
                input_size: 输入特征维度
                num_centers: RBF中心点数量
                sigma: 径向基函数的宽度参数
            """
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
            """
            前向传播
            Args:
                x: 输入特征 [batch_size, input_size]
            Returns:
                suspiciousness: 可疑度预测 [batch_size, input_size]
            """
            # RBF层
            rbf_output = self.rbf_function(x, self.centers)  # [batch_size, num_centers]

            # 线性输出层
            output = self.linear(rbf_output)  # [batch_size, input_size]

            # 应用sigmoid确保输出在[0,1]范围内
            suspiciousness = torch.sigmoid(output)
            return suspiciousness


    class ImprovedSuspiciousnessRBFNN(nn.Module):
        def __init__(self, input_size, num_centers=100, hidden_size=64):
            """
            改进的RBF神经网络，结合了RBF和MLP的优点
            """
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

            # 初始化sigma为合适的值
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
            """前向传播"""
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
        """
        使用K-means初始化RBF中心点
        Args:
            features: 输入特征 [n_samples, input_size]
            num_centers: 中心点数量
        Returns:
            centers: 初始化后的中心点
        """
        features_np = features.cpu().numpy()

        # 使用K-means聚类找到中心点
        kmeans = KMeans(n_clusters=num_centers, random_state=42, n_init=10)
        kmeans.fit(features_np)

        centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
        return centers


    def get_statement_suspiciousness_with_names(model, cov_matrix, stmtmap):
        """获取带有文件名和语句信息的可疑度排名"""
        model.eval()
        with torch.no_grad():
            features = cov_matrix[:, :-1]
            all_predictions = model(features)
            avg_suspiciousness = torch.mean(all_predictions, dim=0)

            # 创建反向映射：索引 -> (文件名, 语句)
            index_to_statement = {idx: key for key, idx in stmtmap.items()}

            # 收集所有语句的可疑度信息
            statement_scores = []
            for idx in range(avg_suspiciousness.shape[0]):
                if idx in index_to_statement:
                    statement_key = index_to_statement[idx]
                    filename, statement = statement_key.split(',', 1)
                    score = avg_suspiciousness[idx].item()
                    statement_scores.append({
                        'filename': filename,
                        'statement': statement,
                        'score': score,
                        'index': idx
                    })

            statement_scores.sort(key=lambda x: x['score'], reverse=True)
            return statement_scores, avg_suspiciousness


    def calculate_function_suspiciousness(statement_scores, failfuncmapstmt):
        """计算函数的平均可疑度"""
        function_scores = {}

        # 创建语句到可疑度的映射
        stmt_to_score = {}
        for stmt_info in statement_scores:
            key = f"{stmt_info['filename']},{stmt_info['statement']}"
            stmt_to_score[key] = stmt_info['score']

        # 计算每个函数的平均可疑度
        for func_name, stmt_set in failfuncmapstmt.items():
            scores = []
            for stmt_key in stmt_set:
                if stmt_key in stmt_to_score:
                    scores.append(stmt_to_score[stmt_key])

            if scores:  # 只计算有对应语句得分的函数
                avg_score = sum(scores) / len(scores)
                function_scores[func_name] = {
                    'avg_score': avg_score,
                    'stmt_count': len(scores),
                    'max_score': max(scores) if scores else 0,
                    'min_score': min(scores) if scores else 0
                }

        # 按平均可疑度排序
        ranked_functions = sorted(function_scores.items(),
                                  key=lambda x: x[1]['avg_score'],
                                  reverse=True)

        return ranked_functions, function_scores


    def calculate_file_suspiciousness(statement_scores):
        """计算文件的平均可疑度"""
        file_scores = defaultdict(list)

        # 按文件分组语句可疑度
        for stmt_info in statement_scores:
            filename = stmt_info['filename']
            file_scores[filename].append(stmt_info['score'])

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


    def complete_training_pipeline(cov_matrix, stmtmap, failfuncmapstmt):
        """完整的训练流程，输出语句、函数和文件级别的可疑度"""
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

        # 确定RBF中心点数量（基于数据大小）
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
                        if covered.sum() > 0:  # 确保有覆盖的语句
                            loss += torch.mean((1 - predictions[i][covered]) ** 2)
                    else:  # 通过测试用例
                        covered = (batch_features[i] == 1)
                        if covered.sum() > 0:  # 确保有覆盖的语句
                            loss += torch.mean(predictions[i][covered] ** 2)

                # 添加正则化项
                reg_loss = 0.001 * torch.norm(model.centers)  # 中心点正则化
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
                # 保存最佳模型
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

        # 获取语句级别的可疑度
        statement_scores, avg_suspiciousness = get_statement_suspiciousness_with_names(model, cov_matrix, stmtmap)

        # 计算函数级别的可疑度
        ranked_functions, function_scores = calculate_function_suspiciousness(statement_scores, failfuncmapstmt)

        # 计算文件级别的可疑度
        ranked_files, file_scores = calculate_file_suspiciousness(statement_scores)

        print("\n训练完成!")

        # 保存结果到文件
        save_detailed_results(statement_scores, ranked_functions, ranked_files, bugId)

        return model, statement_scores, ranked_functions, ranked_files


    def save_detailed_results(statement_scores, ranked_functions, ranked_files, bug_id):
        """保存详细的语句、函数和文件可疑度结果到文件"""
        # 保存语句级别结果
        stmt_output_file = f"suspiciousness_statements_bug_{bug_id}.txt"
        with open(stmt_output_file, 'w', encoding='utf-8') as f:
            f.write(f"语句级别缺陷定位结果 - Bug {bug_id}\n")
            f.write("=" * 100 + "\n")
            f.write(f"{'排名':<6} {'可疑度':<10} {'文件名':<50} {'语句'}\n")
            f.write("=" * 100 + "\n")
            for i, stmt_info in enumerate(statement_scores):
                f.write(
                    f"{i + 1:<6} {stmt_info['score']:<10.6f} {stmt_info['filename']:<50} {stmt_info['statement']}\n")

        # 保存函数级别结果
        func_output_file = f"suspiciousness_functions_bug_{bug_id}.txt"
        with open(func_output_file, 'w', encoding='utf-8') as f:
            f.write(f"函数级别缺陷定位结果 - Bug {bug_id}\n")
            f.write("=" * 120 + "\n")
            f.write(f"{'排名':<6} {'平均可疑度':<12} {'语句数':<8} {'最大可疑度':<12} {'最小可疑度':<12} {'函数名'}\n")
            f.write("=" * 120 + "\n")
            for i, (func_name, score_info) in enumerate(ranked_functions):
                f.write(f"{i + 1:<6} {score_info['avg_score']:<12.6f} {score_info['stmt_count']:<8} "
                        f"{score_info['max_score']:<12.6f} {score_info['min_score']:<12.6f} {func_name}\n")

        # 保存文件级别结果
        file_output_file = f"suspiciousness_files_bug_{bug_id}.txt"
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

        print(f"\n完整结果已保存到:")
        print(f"  语句级别: {stmt_output_file}")
        print(f"  函数级别: {func_output_file}")
        print(f"  文件级别: {file_output_file}")


    # 主程序
    if __name__ == "__main__":
        if cov_matrix is not None and cov_matrix.shape[0] > 1:
            model, statement_scores, ranked_functions, ranked_files = complete_training_pipeline(cov_matrix, stmtmap,
                                                                                                 failfuncmapstmt)
            # 统计信息
            print(f"\n统计信息:")
            print(f"总函数数: {len(ranked_functions)}")
            print(f"总文件数: {len(ranked_files)}")

            # RBF网络特定信息
            print(f"RBF中心点数量: {model.num_centers}")
            print(f"平均sigma值: {model.sigma.mean().item():.4f}")
        else:
            print("错误: 数据不足，无法训练模型")