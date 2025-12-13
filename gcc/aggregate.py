import csv
import os
import random
from collections import defaultdict
from configparser import ConfigParser

# 默认分数范围：GNN/SBFL（分数越高越可疑）和 LLM（分数越高越可疑）
GNN_SBFL_SCORE_RANGE = (-1, 1)  # GNN/SBFL的分数通常在-1到1或0到1，此处使用较宽泛的范围
LLM_SCORE_RANGE = (0, 10)  # LLM的分数通常是0到10


def read_gnn_sbfl_ranking(file_path, score_range=GNN_SBFL_SCORE_RANGE):
    """
    读取 GNN/SBFL 排名文件，返回 {文件名: 归一化分数} 字典 (0-1)
    :param file_path: 文件路径
    :param score_range: 分数原始范围 (min_val, max_val)
    """
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 兼容 GNN 的 'File' 和 SBFL 的文件名格式
            file_name = row.get('File') or row.get('Filename')
            if not file_name: continue

            score = float(row['Score'])
            min_val, max_val = score_range

            # 归一化到 0-1
            normalized_score = (score - min_val) / (max_val - min_val)
            ranking[file_name] = max(0.0, min(1.0, normalized_score))
    return ranking


def read_llm_ranking(file_path, score_range=LLM_SCORE_RANGE):
    """读取 LLM 排名文件（CSV，0-10范围），返回 {文件名: 归一化分数} 字典（0-10→0-1）"""
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['Filename']
            score = float(row['Score'])
            min_val, max_val = score_range

            # 归一化到 0-1
            normalized_score = (score - min_val) / (max_val - min_val)
            ranking[file_name] = max(0.0, min(1.0, normalized_score))
    return ranking


def align_rankings(rankings):
    """对齐多个排名的文件集合，缺失文件补0分"""
    all_files = set()
    for rank in rankings:
        all_files.update(rank.keys())
    aligned = []
    for rank in rankings:
        aligned_rank = {f: rank.get(f, 0.0) for f in all_files}
        aligned.append(aligned_rank)
    return aligned, all_files


def monte_carlo_sample(ranking, n_samples=500, sample_ratio=0.8):
    """对单个排名进行蒙特卡洛采样，生成n_samples个子排名"""
    files = list(ranking.keys())
    total = len(files)
    # 确保采样大小合理
    sample_size = max(1, int(total * sample_ratio)) if total > 0 else 0
    sub_rankings = []

    if sample_size == 0:
        return sub_rankings

    for _ in range(n_samples):
        # 随机抽取文件子集
        sampled_files = random.sample(files, sample_size)

        # 按原分数降序排序（分数高的排前），子排名只包含被采样的文件
        sampled_ranking = sorted(
            sampled_files,
            key=lambda f: ranking[f],
            reverse=True
        )
        sub_rankings.append(sampled_ranking)
    return sub_rankings


def aggregate_sub_rankings(*subs, method="borda"):
    """聚合单轮采样的多个子排名（支持任意数量的子排名）"""
    all_files = set()
    for sub in subs:
        all_files.update(sub)
    scores = defaultdict(float)

    if method == "borda":
        # Borda计数：每个子排名中，排名越前得分越高（得分=子排名长度-索引）
        for sub in subs:
            sub_len = len(sub)
            for idx, f in enumerate(sub):
                scores[f] += (sub_len - idx)

    elif method == "weighted_score":
        # 等权重平均：每个子排名的相对分数（0-1）平均
        for f in all_files:
            total = 0.0
            count = 0
            for sub in subs:
                try:
                    # 查找文件在子排名中的索引
                    s = sub.index(f)
                    # 将索引转换为 0-1 的相对分数（越小越好 -> 越大越好）
                    score = (len(sub) - s) / len(sub)
                    total += score
                    count += 1
                except ValueError:
                    # 文件不在这个子排名中，跳过
                    pass
            scores[f] = total / count if count > 0 else 0.0

    else:
        raise ValueError("聚合方法支持 'borda' 或 'weighted_score'")

    # 按总得分降序排序
    return sorted(scores.keys(), key=lambda f: scores[f], reverse=True)


def final_aggregate(all_sub_aggregates):
    """聚合所有子排名，计算每个文件的平均排名"""
    file_ranks = defaultdict(list)
    for sub in all_sub_aggregates:
        for rank, f in enumerate(sub, 1):  # 排名从1开始
            file_ranks[f].append(rank)

    # 计算平均排名（越小越好）
    avg_ranks = {f: sum(ranks) / len(ranks) for f, ranks in file_ranks.items()}

    # 按平均排名升序排序，得到最终排名
    final_ranking = sorted(avg_ranks.keys(), key=lambda f: avg_ranks[f])
    return final_ranking, avg_ranks


def monte_carlo_aggregate(ranking_paths, output_path,
                          n_samples=500, sample_ratio=0.8, method="borda"):
    """
    蒙特卡洛排序聚合主函数
    :param ranking_paths: {模型名: 文件路径} 字典，指定要聚合的排名文件
    """
    # 1. 读取所有需要的排名文件
    included_rankings = []
    llm_range = LLM_SCORE_RANGE
    gnn_sbfl_range = GNN_SBFL_SCORE_RANGE

    for model_name, path in ranking_paths.items():
        if model_name.lower() == 'llm':
            included_rankings.append(read_llm_ranking(path, llm_range))
        elif model_name.lower() in ['gnn', 'sbfl']:
            included_rankings.append(read_gnn_sbfl_ranking(path, gnn_sbfl_range))
        else:
            print(f"警告: 发现不支持的模型类型 '{model_name}'，已忽略。")

    if not included_rankings:
        print("错误: 没有指定任何有效的排名文件进行聚合。")
        return

    # 2. 对齐所有包含的排名（缺失文件补0分）
    aligned_rankings, all_files = align_rankings(included_rankings)
    print(f"聚合的模型数量: {len(included_rankings)}, 对齐后文件总数: {len(all_files)}")

    # 3. 对每个包含的排名进行蒙特卡洛采样
    print(f"开始蒙特卡洛采样（{n_samples}次，采样比例{sample_ratio}）...")
    subs_list = [monte_carlo_sample(rank, n_samples, sample_ratio)
                 for rank in aligned_rankings]

    # 4. 逐轮聚合子排名
    print(f"开始子排名聚合（方法：{method}）...")
    all_sub_aggregates = []
    for i in range(n_samples):
        # 收集当前轮次每个排名的子排名
        current_subs = [subs[i] for subs in subs_list]
        agg_sub = aggregate_sub_rankings(*current_subs, method=method)
        all_sub_aggregates.append(agg_sub)

    # 5. 最终聚合并输出
    print("计算最终排名...")
    final_ranking, avg_ranks = final_aggregate(all_sub_aggregates)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "File", "AverageRank"])
        for rank, f in enumerate(final_ranking, 1):
            writer.writerow([rank, f, round(avg_ranks[f], 4)])

    print(f"聚合完成，结果已保存至 {output_path}")


def aggregate(bugid, configPath, methods_to_include):
    """
    灵活的聚合主入口函数。

    :param bugid: Bug ID
    :param configPath: 配置文件的路径
    :param methods_to_include: 要聚合的模型列表，例如 ['gnn', 'sbfl'] 或 ['gnn', 'sbfl', 'llm']
    """
    cfg = ConfigParser()
    cfg.read(configPath)
    baseResultDir = cfg.get('gcc-locations', 'resultFile')
    logBaseDir = cfg.get('gcc-locations', 'logDir')

    # 定义所有模型的文件路径
    file_map = {
        'gnn': baseResultDir + bugid + '/result_gnn.csv',
        'sbfl': baseResultDir + bugid + '/resultFile_file.csv',
        'llm': logBaseDir + bugid + '/result_llm.csv'
    }

    # 1. 确定要聚合的文件路径字典
    ranking_paths = {}
    for method in methods_to_include:
        method = method.lower()
        if method in file_map:
            ranking_paths[method] = file_map[method]
        else:
            print(f"警告: 忽略未识别的聚合方法: {method}")

    # 2. 定义输出文件名
    # 根据包含的方法生成动态输出文件名，例如 aggregate_gnn_sbfl.csv
    methods_str = "_".join(sorted(ranking_paths.keys()))
    resulefile = os.path.join(baseResultDir, bugid, f'aggregate_{methods_str}.csv')

    # 3. 调用蒙特卡洛聚合
    monte_carlo_aggregate(
        ranking_paths=ranking_paths,
        output_path=resulefile,
        n_samples=1000,
        sample_ratio=0.85,
        method="borda"
    )

# 示例调用 (在实际应用中，您会从其他地方调用这个函数):
# aggregate(bugid='bug_1', configPath='path/to/config.ini', methods_to_include=['gnn', 'llm'])