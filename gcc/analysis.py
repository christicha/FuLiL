from configparser import ConfigParser
import os
import csv
from typing import List, Dict, Union, Any


# ---------------------------------------------
# 辅助函数：读取缺陷文件列表和获取排名 (保持不变)
# ---------------------------------------------

def get_buggy_names(location_file_path: str) -> List[str]:
    # ... (此函数保持不变) ...
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


def get_rank(file_path: str, buggy_name: str, rank_col_name: str = 'Rank') -> Union[str, int, float]:
    # ... (此函数保持不变) ...
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
            file_col_name = None
            if 'File' in reader.fieldnames:
                file_col_name = 'File'
            elif 'Filename' in reader.fieldnames:
                file_col_name = 'Filename'
            else:
                return 'N/A'

            # 确定排名所在的列名
            current_rank_col = None
            if rank_col_name in reader.fieldnames:
                current_rank_col = rank_col_name
            elif 'AverageRank' in reader.fieldnames:
                current_rank_col = 'AverageRank'
            elif 'Rank' in reader.fieldnames:
                current_rank_col = 'Rank'
            else:
                return 'N/A'

            for row in reader:
                if row.get(file_col_name) == buggy_name:
                    rank_value = row.get(current_rank_col)
                    if rank_value is not None:
                        try:
                            # 尝试转换为整数或浮点数
                            if current_rank_col == 'Rank':
                                return int(float(rank_value))
                            else:
                                return float(rank_value)
                        except ValueError:
                            return rank_value
            return 'N/A'
    except Exception as e:
        # print(f"读取排名文件 {os.path.basename(file_path)} 时发生错误: {e}")
        return 'N/A'


# ---------------------------------------------
# 主分析函数：只收集排名数据，不写入文件
# ---------------------------------------------

def collect_ranks(bugid: str, configPath: str, aggregate_file_path: str, agg_suffix: str, llm_k: int,
                  all_files_map: Dict[str, Dict[str, Any]]) -> None:
    """
    收集单个 Bug ID 下，独立模型和特定聚合模型的排名结果，并更新到 all_files_map 中。

    :param bugid: 缺陷 ID。
    :param configPath: 配置文件的路径。
    :param aggregate_file_path: 待分析的聚合文件路径。
    :param agg_suffix: 用于作为聚合排名的列名后缀。
    :param llm_k: LLM 文件的 k 值后缀。
    :param all_files_map: 存储所有排名数据的字典（{文件路径: {模型: 排名}}）。
    """
    cfg = ConfigParser()
    cfg.read(configPath)


    baseResultDir = cfg.get('gcc-locations', 'resultFile')
    logBaseDir = cfg.get('gcc-locations', 'logDir')
    infoResultDir = cfg.get('gcc-locations', 'infodir')

    location = os.path.join(infoResultDir, bugid, 'locations')

    # 独立模型文件路径
    GNNfile = os.path.join(baseResultDir, bugid, 'result_gnn.csv')
    SBFLfile = os.path.join(baseResultDir, bugid, 'resultFile_file.csv')
    # LLM 文件名现在是带 k 值的
    LLMfile = os.path.join(logBaseDir, bugid, f'result_llm_{llm_k}.csv')

    # 1. 获取所有缺陷文件名称
    buggy_names = get_buggy_names(location)
    if not buggy_names:
        return

    # 2. 逐个缺陷文件获取排名
    for buggy_name in buggy_names:
        # 确保每个缺陷文件在 Map 中有记录
        if buggy_name not in all_files_map:
            all_files_map[buggy_name] = {}

        # --- 独立模型排名 (只在第一次运行时读取) ---
        if 'SBFL_Rank' not in all_files_map[buggy_name]:
            all_files_map[buggy_name]['SBFL_Rank'] = get_rank(SBFLfile, buggy_name, 'Rank')
            all_files_map[buggy_name]['GNN_Rank'] = get_rank(GNNfile, buggy_name, 'Rank')
            all_files_map[buggy_name]['LLM_Rank'] = get_rank(LLMfile, buggy_name, 'Rank')

        # --- 聚合模型排名 (每次都读取) ---
        agg_rank_key = f"{agg_suffix}_Agg_Rank"
        all_files_map[buggy_name][agg_rank_key] = get_rank(aggregate_file_path, buggy_name, 'Rank')


# ---------------------------------------------
# 运行所有配置的包装函数 (执行收集和写入)
# ---------------------------------------------

def analysis(bugid: str, revision: str, configPath: str, k):
    """
    定义三种聚合配置，收集所有排名数据，并写入单个文件。

    :param llm_k: LLM 文件的 k 值后缀。
    """
    cfg = ConfigParser()
    cfg.read(configPath)

    # 统一使用 gcc-locations 配置中的路径来定义文件
    baseResultDir = cfg.get('gcc-locations', 'resultFile')

    # 1. 定义三种聚合的文件路径和列名后缀
    aggregation_configs: Dict[str, str] = {
        # 后缀(agg_suffix) : 完整文件路径
        'GNN_SBFL': os.path.join(baseResultDir, bugid, f'aggregate_gnn_sbfl_{k}.csv'),
        'SBFL_LLM': os.path.join(baseResultDir, bugid, f'aggregate_llm_sbfl_{k}.csv'),
        'GNN_LLM': os.path.join(baseResultDir, bugid, f'aggregate_gnn_llm_{k}.csv'),
        'GNN_SBFL_LLM': os.path.join(baseResultDir, bugid, f'aggregate_gnn_llm_sbfl_{k}.csv')
    }

    # 2. 初始化数据收集字典
    all_files_map: Dict[str, Dict[str, Any]] = {}

    # 3. 逐个配置收集排名数据
    print(f"--- 开始收集 Bug ID {bugid} 的所有排名数据 (LLM k={k}) ---")
    for suffix, agg_file_path in aggregation_configs.items():
        # ⚠️ 关键检查：打印文件路径，确保路径正确
        print(f"  > 检查聚合文件: {suffix} -> {agg_file_path}")

        # 只有文件存在时才进行收集
        if not os.path.exists(agg_file_path):
            print(f"  > 警告: 文件不存在，跳过聚合配置 {suffix}")
            # 确保即使文件不存在，该键也被添加到 map 中，值为 'N/A'
            # 这一步是为了保证后续写入 CSV 时 fieldnames 不会丢失，并保持 N/A 记录
            buggy_names = get_buggy_names(os.path.join(cfg.get('gcc-locations', 'infodir'), bugid, 'locations'))
            if buggy_names:
                agg_rank_key = f"{suffix}_Agg_Rank"
                for buggy_name in buggy_names:
                    if buggy_name not in all_files_map:
                        all_files_map[buggy_name] = {}
                    all_files_map[buggy_name][agg_rank_key] = 'N/A'
            continue  # 跳过 collect_ranks

        collect_ranks(bugid, configPath, agg_file_path, suffix, k, all_files_map)
        print(f"  > 收集聚合配置 {suffix} 完成。")

    if not all_files_map:
        print("未收集到任何排名数据，分析终止。")
        return

    # 4. 准备最终输出 CSV 文件名
    final_analysis_csv = os.path.join(baseResultDir, bugid, f'analysis_combined.csv')
    os.makedirs(os.path.dirname(final_analysis_csv), exist_ok=True)

    # 5. 确定最终 CSV 的列名顺序
    fixed_headers = ["File", "SBFL_Rank", "GNN_Rank", "LLM_Rank"]
    agg_headers = [f"{suffix}_Agg_Rank" for suffix in aggregation_configs.keys()]
    final_headers = fixed_headers + agg_headers

    # 6. 写入最终的合并 CSV 文件
    with open(final_analysis_csv, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=final_headers)
        writer.writeheader()

        print("\n--- 最终合并结果 ---")
        for file_name, ranks_data in all_files_map.items():
            # 创建要写入的字典行
            row_to_write = {'File': file_name}
            # 使用 update 确保所有键都被加入
            row_to_write.update(ranks_data)

            # 格式化输出到控制台
            log_output = f"结果：{file_name} | "
            for h in fixed_headers[1:] + agg_headers:
                # 修正日志输出的键名，去掉 '_Rank' 和 '_Agg_Rank' 后缀，保持简洁
                display_key = h.replace('_Agg_Rank', '_Agg').replace('_Rank', '')
                log_output += f"{display_key}: {row_to_write.get(h, 'N/A')} | "
            print(log_output.rstrip(' | '))

            # 确保所有列都有值，避免 DictWriter 报错（虽然 fieldnames 应该能处理）
            final_row = {h: row_to_write.get(h, 'N/A') for h in final_headers}
            writer.writerow(final_row)

    print(f"\n✅ 所有分析结果已成功合并并保存至：{final_analysis_csv}")

