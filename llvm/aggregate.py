import csv
import random
from collections import defaultdict
from configparser import ConfigParser


def read_rbfnn_sbfl_ranking(file_path):
    """读取RBFNN/SBFL排名文件（CSV，0-1范围），返回 {文件名: 分数} 字典"""
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['File']  # RBFNN/SBFL的文件名列是'File'
            score = float(row['Score'])
            # 确保分数在0-1范围内（按需求）
            ranking[file_name] = max(0.0, min(1.0, score))
    return ranking


def read_llm_ranking(file_path):
    """读取LLM排名文件（CSV，0-10范围），返回 {文件名: 归一化分数} 字典（0-10→0-1）"""
    ranking = {}
    with open(file_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row['Filename']  # LLM的文件名列是'Filename'
            score = float(row['Score'])
            # 归一化到0-1范围（原范围0-10）
            normalized_score = max(0.0, min(10.0, score)) / 10.0  # 限制异常值后归一化
            ranking[file_name] = normalized_score
    return ranking


def align_three_rankings(rbfnn, sbfl, llm):
    """对齐三个排名的文件集合，缺失文件补0分"""
    all_files = set(rbfnn.keys()).union(sbfl.keys()).union(llm.keys())
    aligned_rbfnn = {f: rbfnn.get(f, 0.0) for f in all_files}
    aligned_sbfl = {f: sbfl.get(f, 0.0) for f in all_files}
    aligned_llm = {f: llm.get(f, 0.0) for f in all_files}
    return aligned_rbfnn, aligned_sbfl, aligned_llm


def monte_carlo_sample(ranking, n_samples=500, sample_ratio=0.8):
    """对单个排名进行蒙特卡洛采样，生成n_samples个子排名"""
    files = list(ranking.keys())
    total = len(files)
    sample_size = max(1, int(total * sample_ratio))  # 采样大小（至少1个文件）
    sub_rankings = []

    for _ in range(n_samples):
        # 无放回采样，保留排名特征
        sampled_files = random.sample(files, sample_size)
        # 按原分数降序排序（分数高的排前）
        sampled_ranking = sorted(
            sampled_files,
            key=lambda f: ranking[f],
            reverse=True
        )
        sub_rankings.append(sampled_ranking)

    return sub_rankings


def aggregate_sub_rankings(sub1, sub2, sub3, method="borda"):
    """聚合单轮采样的三个子排名（分别来自RBFNN、SBFL、LLM）"""
    all_files = set(sub1).union(sub2).union(sub3)
    scores = defaultdict(float)

    if method == "borda":
        # Borda计数：每个子排名中，排名越前得分越高（得分=子排名长度-索引）
        len1, len2, len3 = len(sub1), len(sub2), len(sub3)
        for idx, f in enumerate(sub1):
            scores[f] += (len1 - idx)  # RBFNN子排名得分
        for idx, f in enumerate(sub2):
            scores[f] += (len2 - idx)  # SBFL子排名得分
        for idx, f in enumerate(sub3):
            scores[f] += (len3 - idx)  # LLM子排名得分
    elif method == "weighted_score":
        # 加权分数求和（三排名等权重，各占1/3）
        for f in all_files:
            # 从子排名中反查相对分数（越前分数越高）
            s1 = sub1.index(f) if f in sub1 else -1
            s2 = sub2.index(f) if f in sub2 else -1
            s3 = sub3.index(f) if f in sub3 else -1

            # 相对分数计算（归一化到0-1）
            score1 = (len(sub1) - s1) / len(sub1) if s1 != -1 else 0.0
            score2 = (len(sub2) - s2) / len(sub2) if s2 != -1 else 0.0
            score3 = (len(sub3) - s3) / len(sub3) if s3 != -1 else 0.0

            # 等权重融合（1/3 each）
            scores[f] = (score1 + score2 + score3) / 3.0
    else:
        raise ValueError("聚合方法支持 'borda' 或 'weighted_score'")

    # 按总得分降序排序，返回当前轮的聚合子排名
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


def monte_carlo_aggregate(rbfnn_path, sbfl_path, llm_path, output_path,
                          n_samples=500, sample_ratio=0.8, method="borda"):
    """
    蒙特卡洛排序聚合主函数（支持三个排名文件）
    :param rbfnn_path: RBFNN排名文件路径（CSV，0-1）
    :param sbfl_path: SBFL排名文件路径（CSV，0-1）
    :param llm_path: LLM排名文件路径（CSV，0-10）
    :param output_path: 输出聚合结果的路径
    :param n_samples: 蒙特卡洛采样次数
    :param sample_ratio: 每次采样的文件比例
    :param method: 子排名聚合方法（borda/weighted_score）
    """
    # 1. 读取三个排名文件（并对LLM做归一化）
    rbfnn_ranking = read_rbfnn_sbfl_ranking(rbfnn_path)
    sbfl_ranking = read_rbfnn_sbfl_ranking(sbfl_path)
    llm_ranking = read_llm_ranking(llm_path)

    # 2. 对齐三个排名（缺失文件补0分）
    aligned_rbfnn, aligned_sbfl, aligned_llm = align_three_rankings(
        rbfnn_ranking, sbfl_ranking, llm_ranking
    )
    print(f"对齐后文件总数: {len(aligned_rbfnn)}")

    # 3. 对三个排名分别进行蒙特卡洛采样
    print(f"开始蒙特卡洛采样（{n_samples}次，采样比例{sample_ratio}）...")
    rbfnn_subs = monte_carlo_sample(aligned_rbfnn, n_samples, sample_ratio)
    sbfl_subs = monte_carlo_sample(aligned_sbfl, n_samples, sample_ratio)
    llm_subs = monte_carlo_sample(aligned_llm, n_samples, sample_ratio)

    # 4. 逐轮聚合三个子排名
    print(f"开始子排名聚合（方法：{method}）...")
    all_sub_aggregates = []
    for r_sub, s_sub, l_sub in zip(rbfnn_subs, sbfl_subs, llm_subs):
        agg_sub = aggregate_sub_rankings(r_sub, s_sub, l_sub, method)
        all_sub_aggregates.append(agg_sub)

    # 5. 最终聚合，计算平均排名
    print("计算最终排名...")
    final_ranking, avg_ranks = final_aggregate(all_sub_aggregates)

    # 6. 输出结果（排名、文件、平均排名）
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Rank", "File", "AverageRank"])
        for rank, f in enumerate(final_ranking, 1):
            writer.writerow([rank, f, round(avg_ranks[f], 4)])
    print(f"聚合完成，结果已保存至 {output_path}")


def aggregate(bugid, configPath):
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
        n_samples=1000,  # 采样次数
        sample_ratio=0.95,  # 每次采样95%的文件
        method="borda"  # 聚合方法（borda或weighted_score）
    )