from configparser import ConfigParser
import os
import csv
from typing import List, Dict, Union


# ---------------------------------------------
# 辅助函数：读取缺陷文件列表和获取排名
# ---------------------------------------------

def get_buggy_names(location_file_path: str) -> List[str]:
    """从 location 文件中解析出所有缺陷文件的路径。"""
    buggy_names: List[str] = []

    if not os.path.exists(location_file_path):
        print(f"错误: 缺陷位置文件不存在: {location_file_path}")
        return []

    try:
        with open(location_file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"读取缺陷位置文件失败: {e}")
        return []

    buggy_keywords = ['buggy location', 'buggy locations', 'buggy locations:', 'buggy location:']
    buggy_idx = -1

    # 查找关键词的索引
    for keyword in buggy_keywords:
        if keyword in lines:
            buggy_idx = lines.index(keyword)
            break

    if buggy_idx == -1:
        print(f"警告: 在文件 {location_file_path} 中未找到缺陷位置关键词。")
        return []

    # 从关键词下一行开始解析文件路径
    for line in lines[buggy_idx + 1:]:
        if line.startswith('file:'):
            # 提取文件路径并去除不必要的字符串
            try:
                full_path = line.split('file:')[1].split(';')[0].strip()
                # 移除 'gcc/', 'trunk/'
                buggy_name = full_path.replace('gcc/', '').replace('trunk/', '')

                # 处理开头的斜杠
                buggy_name = buggy_name.lstrip()
                if buggy_name.startswith('/'):
                    buggy_name = buggy_name[1:]

                if buggy_name and buggy_name not in buggy_names:
                    buggy_names.append(buggy_name)
            except IndexError:
                # 忽略格式不正确的行
                continue

    return buggy_names


def get_rank(file_path: str, buggy_name: str, rank_col_name: str = 'Rank') -> Union[str, int]:
    """
    从排名 CSV 文件中查找特定缺陷文件的排名。

    :param file_path: 排名 CSV 文件路径。
    :param buggy_name: 缺陷文件名称。
    :param rank_col_name: 排名所在列的名称，默认为 'Rank' 或 'AverageRank'。
    :return: 排名数字或 'N/A'。
    """
    if not os.path.exists(file_path):
        return 'N/A'

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # 确定文件名所在的列名
            # 'File' 用于 GNN/SBFL/Aggregate, 'Filename' 用于 LLM
            file_col_name = None
            if 'File' in reader.fieldnames:
                file_col_name = 'File'
            elif 'Filename' in reader.fieldnames:
                file_col_name = 'Filename'
            else:
                # 无法识别文件名列
                return 'N/A'

            # 确定排名所在的列名 (通常是第一列，但为了通用性，我们查找列名)
            if rank_col_name not in reader.fieldnames:
                # 聚合文件可能使用 AverageRank
                if 'AverageRank' in reader.fieldnames:
                    rank_col_name = 'AverageRank'
                elif 'Rank' in reader.fieldnames:
                    rank_col_name = 'Rank'
                else:
                    return 'N/A'

            for row in reader:
                # 查找匹配的行
                if row.get(file_col_name) == buggy_name:
                    rank_value = row.get(rank_col_name)
                    if rank_value is not None:
                        # 对于聚合结果的平均排名，我们返回浮点数，否则返回整数排名
                        try:
                            return int(float(rank_value)) if rank_col_name == 'Rank' else float(rank_value)
                        except ValueError:
                            return rank_value  # 返回原始字符串
            return 'N/A'
    except Exception as e:
        print(f"读取排名文件 {os.path.basename(file_path)} 时发生错误: {e}")
        return 'N/A'


# ---------------------------------------------
# 主分析函数 (针对单个聚合文件)
# ---------------------------------------------

def analysis(bugid: str, configPath: str, aggregate_file_path: str, output_suffix: str):
    """
    分析单个 Bug ID 下 GNN/SBFL/LLM/特定聚合模型的排名结果。

    :param bugid: 缺陷 ID。
    :param configPath: 配置文件的路径。
    :param aggregate_file_path: 待分析的聚合文件路径。
    :param output_suffix: 用于区分输出分析文件的后缀（如 'gnn_sbfl'）。
    """
    cfg = ConfigParser()
    cfg.read(configPath)

    infoResultDir = cfg.get('gcc-locations', 'infodir')
    baseResultDir = cfg.get('gcc-locations', 'resultFile')
    logBaseDir = cfg.get('gcc-locations', 'logDir')

    location = os.path.join(infoResultDir, bugid, 'locations')

    # 独立模型文件路径
    GNNfile = os.path.join(baseResultDir, bugid, 'result_gnn.csv')
    SBFLfile = os.path.join(baseResultDir, bugid, 'resultFile_file.csv')
    LLMfile = os.path.join(logBaseDir, bugid, 'result_llm.csv')

    # 最终的分析输出文件
    analysis_csv = os.path.join(baseResultDir, bugid, f'analysis_{output_suffix}.csv')

    # 1. 获取所有缺陷文件名称
    buggy_names = get_buggy_names(location)
    if not buggy_names:
        print(f"Bug ID {bugid}：未找到缺陷文件列表，跳过分析。")
        return

    print(f"\n--- 开始分析 Bug ID {bugid}，聚合配置: {output_suffix} ---")

    # 2. 准备写入分析结果
    os.makedirs(os.path.dirname(analysis_csv), exist_ok=True)

    with open(analysis_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 写入 CSV 头部
        writer.writerow(["File", "SBFL_Rank", "GNN_Rank", "LLM_Rank", f"{output_suffix}_Agg_Rank"])

        # 3. 逐个缺陷文件获取排名
        for buggy_name in buggy_names:
            # 独立模型排名 (Rank 列是整数)
            sbfl_rank = get_rank(SBFLfile, buggy_name, 'Rank')
            gnn_rank = get_rank(GNNfile, buggy_name, 'Rank')
            llm_rank = get_rank(LLMfile, buggy_name, 'Rank')

            # 聚合模型排名 (AverageRank 列可能是浮点数)
            aggregate_rank = get_rank(aggregate_file_path, buggy_name, 'AverageRank')

            # 写入 CSV 行
            writer.writerow([buggy_name, sbfl_rank, gnn_rank, llm_rank, aggregate_rank])

            print(
                f"结果：{buggy_name} | SBFL排名：{sbfl_rank} | GNN排名：{gnn_rank} | LLM排名：{llm_rank} | {output_suffix}排名：{aggregate_rank}")

    print(f"分析结果已保存至：{analysis_csv}")


# ---------------------------------------------
# 运行所有配置的包装函数
# ---------------------------------------------

def run_analysis_for_all_configs(bugid: str, revision: str, configPath: str):
    """定义并运行三种聚合配置的分析。"""
    cfg = ConfigParser()
    cfg.read(configPath)
    baseResultDir = cfg.get('gcc-locations', 'resultFile')

    # 定义三种聚合的文件名和对应的后缀
    aggregation_configs: Dict[str, str] = {
        'gnn_sbfl': os.path.join(baseResultDir, bugid, 'aggregate_gnn_sbfl.csv'),
        'sbfl_llm': os.path.join(baseResultDir, bugid, 'aggregate_sbfl_llm.csv'),
        'gnn_sbfl_llm': os.path.join(baseResultDir, bugid, 'aggregate_gnn_sbfl_llm.csv')
    }

    for suffix, agg_file_path in aggregation_configs.items():
        analysis(bugid, configPath, agg_file_path, suffix)

