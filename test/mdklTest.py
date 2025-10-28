import csv
import random
from collections import defaultdict


def read_ranking_file(file_path):
    """读取排名文件，返回 {文件名: 分数} 字典"""
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['File']
            score = float(row['Score'])
            ranking[file_name] = score
    return ranking


def align_rankings(ranking1, ranking2):
    """对齐两个排名的文件集合，缺失文件补0分"""
    all_files = set(ranking1.keys()).union(set(ranking2.keys()))
    aligned1 = {f: ranking1.get(f, 0.0) for f in all_files}
    aligned2 = {f: ranking2.get(f, 0.0) for f in all_files}
    return aligned1, aligned2


def monte_carlo_sample(ranking, n_samples=500, sample_ratio=0.8):
    """对单个排名进行蒙特卡洛采样，生成n_samples个子排名"""
    files = list(ranking.keys())
    total = len(files)
    sample_size = max(1, int(total * sample_ratio))  # 采样大小（至少1个文件）
    sub_rankings = []

    for _ in range(n_samples):
        # 有放回采样（允许重复，但实际中为保留排名特征，用无放回采样）
        sampled_files = random.sample(files, sample_size)
        # 按原分数降序排序，生成子排名（分数高的排前）
        sampled_ranking = sorted(
            sampled_files,
            key=lambda f: ranking[f],
            reverse=True
        )
        sub_rankings.append(sampled_ranking)

    return sub_rankings


def aggregate_sub_rankings(sub1, sub2, method="borda"):
    """聚合单轮采样的两个子排名（sub1来自R1，sub2来自R2）"""
    # 合并两个子排名的文件集合
    all_files = set(sub1).union(set(sub2))
    scores = defaultdict(float)

    if method == "borda":
        # Borda计数：排名越前得分越高（得分=总文件数-排名索引）
        len1, len2 = len(sub1), len(sub2)
        for idx, f in enumerate(sub1):
            scores[f] += (len1 - idx)  # sub1的Borda得分
        for idx, f in enumerate(sub2):
            scores[f] += (len2 - idx)  # sub2的Borda得分
    elif method == "weighted_score":
        # 加权分数求和（直接用原分数加权，这里权重1:1）
        for f in all_files:
            s1 = sub1.index(f) if f in sub1 else -1  # 若不在子排名中，分数视为0
            s2 = sub2.index(f) if f in sub2 else -1
            # 从子排名中反查原分数（这里简化为用排名位置映射分数，越前分数越高）
            score1 = (len(sub1) - s1) / len(sub1) if s1 != -1 else 0.0
            score2 = (len(sub2) - s2) / len(sub2) if s2 != -1 else 0.0
            scores[f] = 0.5 * score1 + 0.5 * score2  # 等权重融合
    else:
        raise ValueError("聚合方法支持 'borda' 或 'weighted_score'")

    # 按得分降序排序，返回当前轮的聚合子排名
    return sorted(scores.keys(), key=lambda f: scores[f], reverse=True)


def final_aggregate(all_sub_aggregates):
    """聚合所有子排名，计算每个文件的平均排名"""
    file_ranks = defaultdict(list)
    total_sub = len(all_sub_aggregates)

    for sub in all_sub_aggregates:
        for rank, f in enumerate(sub, 1):  # 排名从1开始
            file_ranks[f].append(rank)

    # 计算平均排名（越小越好）
    avg_ranks = {
        f: sum(ranks) / len(ranks)
        for f, ranks in file_ranks.items()
    }

    # 按平均排名升序排序，得到最终排名
    final_ranking = sorted(avg_ranks.keys(), key=lambda f: avg_ranks[f])
    return final_ranking, avg_ranks


def monte_carlo_aggregate(rbfnn_path, sbfl_path, output_path,
                          n_samples=500, sample_ratio=0.8, method="borda"):
    """
    蒙特卡洛排序聚合主函数
    :param rbfnn_path: RBFNN排名文件路径
    :param sbfl_path: SBFL排名文件路径
    :param output_path: 输出聚合结果的路径
    :param n_samples: 蒙特卡洛采样次数
    :param sample_ratio: 每次采样的文件比例
    :param method: 子排名聚合方法（borda/weighted_score）
    """
    # 1. 读取并对齐两个排名
    rbfnn_ranking = read_ranking_file(rbfnn_path)
    sbfl_ranking = read_ranking_file(sbfl_path)
    aligned_rbfnn, aligned_sbfl = align_rankings(rbfnn_ranking, sbfl_ranking)
    print(f"对齐后文件总数: {len(aligned_rbfnn)}")

    # 2. 对两个排名分别进行蒙特卡洛采样
    print(f"开始蒙特卡洛采样（{n_samples}次，采样比例{sample_ratio}）...")
    rbfnn_subs = monte_carlo_sample(aligned_rbfnn, n_samples, sample_ratio)
    sbfl_subs = monte_carlo_sample(aligned_sbfl, n_samples, sample_ratio)

    # 3. 逐轮聚合子排名
    print(f"开始子排名聚合（方法：{method}）...")
    all_sub_aggregates = []
    for r_sub, s_sub in zip(rbfnn_subs, sbfl_subs):
        agg_sub = aggregate_sub_rankings(r_sub, s_sub, method)
        all_sub_aggregates.append(agg_sub)

    # 4. 最终聚合，计算平均排名
    print("计算最终排名...")
    final_ranking, avg_ranks = final_aggregate(all_sub_aggregates)

    # 5. 输出结果（排名、文件、平均排名）
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "File", "AverageRank"])
        for rank, f in enumerate(final_ranking, 1):
            writer.writerow([rank, f, round(avg_ranks[f], 4)])
    print(f"聚合完成，结果已保存至 {output_path}")


# 示例调用（替换为你的文件路径）
if __name__ == "__main__":
    # 输入文件路径（RBFNN和SBFL的排名文件）
    rbfnn_file = "/home/chris/FLL-workplace/llvmbugs/result/16069/NNresultFile_file_with_attention.csv"  # 替换为你的RBFNN排名文件
    sbfl_file = "/home/chris/FLL-workplace/llvmbugs/result/16069/resultFile_file.csv"  # 替换为你的SBFL排名文件
    output_file = "aggregated_ranking.csv"  # 输出聚合结果

    # 运行聚合算法
    monte_carlo_aggregate(
        rbfnn_path=rbfnn_file,
        sbfl_path=sbfl_file,
        output_path=output_file,
        n_samples=500,  # 采样次数（建议500-1000）
        sample_ratio=0.8,  # 每次采样80%的文件
        method="borda"  # 聚合方法（borda或weighted_score）
    )