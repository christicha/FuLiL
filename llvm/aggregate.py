import csv
import random
from collections import defaultdict
from configparser import ConfigParser

import csv


def read_rbfnn_sbfl_ranking(file_path, rbfnn_score_range=(-1, 1)):
    """
    读取RBFNN/SBFL排名文件，返回 {文件名: 归一化分数} 字典
    :param file_path: 文件路径
    :param rbfnn_score_range: RBFNN分数原始范围，默认(-1, 1)；SBFL传(0, 1)
    """
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['File']
            score = float(row['Score'])
            min_val, max_val = rbfnn_score_range
            # 归一化到0-1，同时限制异常值
            normalized_score = (score - min_val) / (max_val - min_val)
            ranking[file_name] = max(0.0, min(1.0, normalized_score))
    return ranking


def read_llm_ranking(file_path):
    """读取LLM排名文件（CSV，0-10范围），返回 {文件名: 归一化分数} 字典（0-10→0-1）"""
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['Filename']
            score = float(row['Score'])
            normalized_score = max(0.0, min(10.0, score)) / 10.0  # 归一化到0-1
            ranking[file_name] = normalized_score
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
    sample_size = max(1, int(total * sample_ratio))  # 至少采样1个文件
    sub_rankings = []
    for _ in range(n_samples):
        sampled_files = random.sample(files, sample_size)
        # 按原分数降序排序（分数高的排前）
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
                if f in sub:
                    s = sub.index(f)
                    score = (len(sub) - s) / len(sub)  # 归一化到0-1
                    total += score
                    count += 1
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
    avg_ranks = {f: sum(ranks)/len(ranks) for f, ranks in file_ranks.items()}
    # 按平均排名升序排序，得到最终排名
    final_ranking = sorted(avg_ranks.keys(), key=lambda f: avg_ranks[f])
    return final_ranking, avg_ranks


def monte_carlo_aggregate(rbfnn_path, sbfl_path, llm_path, output_path,
                          n_samples=500, sample_ratio=0.8, method="borda",
                          include_rbfnn=True):  # 是否包含RBFNN
    """
    蒙特卡洛排序聚合主函数（支持控制是否包含RBFNN）
    :param include_rbfnn: 是否将RBFNN排名纳入聚合，默认True
    """
    # 1. 读取需要包含的排名文件
    included_rankings = []
    if include_rbfnn:
        included_rankings.append(read_rbfnn_sbfl_ranking(rbfnn_path))
    included_rankings.append(read_rbfnn_sbfl_ranking(sbfl_path))  # 始终包含SBFL
    included_rankings.append(read_llm_ranking(llm_path))  # 始终包含LLM

    # 2. 对齐所有包含的排名（缺失文件补0分）
    aligned_rankings, all_files = align_rankings(included_rankings)
    print(f"对齐后文件总数: {len(all_files)}")

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


def aggregate(bugid, configPath, include_rbfnn=True):
    cfg = ConfigParser()
    cfg.read(configPath)
    baseResultDir = cfg.get('llvm-locations', 'resultFile')
    logBaseDir = cfg.get('llvm-locations', 'logDir')
    # 三个排名文件路径
    RBFNNfile = baseResultDir + bugid + '/NNresultFile_file_with_attention.csv'
    SBFLfile = baseResultDir + bugid + '/resultFile_file.csv'
    LLMfile = logBaseDir + bugid + '/result_llm.csv'
    resulefile = baseResultDir + bugid + '/aggregate_result.csv'

    monte_carlo_aggregate(
        rbfnn_path=RBFNNfile,
        sbfl_path=SBFLfile,
        llm_path=LLMfile,
        output_path=resulefile,
        n_samples=1000,
        sample_ratio=0.85,
        method="borda",
        include_rbfnn=include_rbfnn
    )