import csv
import os.path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
import time
import re

config = ConfigParser()
config.read('../config/config.ini', encoding='utf-8')
compilerBasePath = config.get('llvm-locations', 'compilersdir')
baseResultPath = config.get('llvm-locations', 'resultFile')
historylist = config.get('llvm-locations', 'historyList')
starttime = '2003-12-18'

bugIds = []
revisions = []
endtimes = []

with open(historylist, 'r') as f:
    for line in f:
        line = line.strip()
        items = line.split(',')
        bugIds.append(items[0])
        revisions.append(items[1])
        endtimes.append(items[2])


def extract_function_name(function_signature):
    """
    从完整的函数签名中提取纯函数名（去掉类名前缀）
    例如: "InnerLoopVectorizer::vectorizeLoop(LoopVectorizationLegality *Legal)"
          -> "vectorizeLoop"
    """
    # 分离函数名和参数部分
    if '(' in function_signature:
        func_part = function_signature.split('(')[0].strip()
    else:
        func_part = function_signature

    # 如果包含 :: 说明是成员函数，取最后一部分
    if '::' in func_part:
        return func_part.split('::')[-1]
    else:
        return func_part


def build_precise_function_regex(function_signature):
    """
    构建精确的函数正则表达式，去掉类名前缀，只匹配函数名
    """
    # 提取纯函数名
    pure_function_name = extract_function_name(function_signature)
    # 分离函数名和参数部分
    if '(' not in function_signature:
        return re.escape(pure_function_name) + r'\s*\(\s*\)'
    # 提取参数部分
    params_part = function_signature.split('(', 1)[1].rstrip(')')
    # 转义函数名
    escaped_func_name = re.escape(pure_function_name)
    # 处理参数部分 - 精确匹配参数类型
    if not params_part.strip():
        # 无参数
        return escaped_func_name + r'\s*\(\s*\)'
    # 分割参数
    params = [p.strip() for p in params_part.split(',')]
    # 为每个参数构建正则表达式
    param_regexes = []
    for param in params:
        if not param:
            continue
        # 处理参数类型和名称
        param_parts = param.split()
        if len(param_parts) >= 2:
            # 有类型和名称，如 "LoopVectorizationLegality *Legal"
            param_type = ' '.join(param_parts[:-1])  # 类型部分
            param_name = param_parts[-1]  # 名称部分

            # 转义类型，但允许空格变化
            escaped_type = re.escape(param_type).replace(r'\ ', r'\s+')
            # 名称部分使用通配符，因为参数名可能变化
            param_regex = escaped_type + r'\s+\w+'
        else:
            # 只有类型，如 "void"
            param_regex = re.escape(param)
        param_regexes.append(param_regex)
    # 构建完整的参数正则
    if param_regexes:
        params_regex = r'\s*' + r'\s*,\s*'.join(param_regexes) + r'\s*'
    else:
        params_regex = r'\s*'
    return escaped_func_name + r'\s*\(' + params_regex + r'\)'


def process_single_function(args):
    rank, function_field, endtime = args
    if ';' in function_field:
        file_path, function_signature = function_field.split(';', 1)
    else:
        print(f"警告: 无法解析函数字段: {function_field}")
    precise_regex = build_precise_function_regex(function_signature)
    try:
        # 构建git命令
        cmd = [
            'git', 'log',
            '--since', starttime,
            '--until', endtime,
            '--follow',
            '--format=%H',
            '-G', precise_regex,
            '--', file_path
        ]
        # 执行命令
        print(cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        count = len([line for line in result.stdout.strip().split('\n') if line.strip()])
    except subprocess.CalledProcessError as e:
        print(f"执行git命令出错: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return 0
    except FileNotFoundError:
        print("git命令未找到，请确保git已安装且在PATH中")
        return 0
    # print(
    #     f"rank: {rank} - function: {precise_regex} - count: {count}"
    # )
    return {
        'Rank': rank,
        'Function': function_field,
        'Count': count
    }

for i in range(len(bugIds)):
    bugid = bugIds[i]
    revision = revisions[i]
    endtime = endtimes[i]
    os.chdir(compilerBasePath + revision + '/' + revision + '/llvm')
    functionPath = baseResultPath + bugid + '/resultFile_func.csv'
    resultPath = baseResultPath + bugid + '/resultFile_modi.csv'
    processed_ranks = set()
    if os.path.exists(resultPath):
        try:
            with open(resultPath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    processed_ranks.add(row['Rank'])
            print(f"发现已存在的输出文件，跳过 {len(processed_ranks)} 个已处理的行")
        except Exception as e:
            print(f"读取现有输出文件时出错: {e}")
    tasks = []
    with open(functionPath, 'r') as f:
        lines = f.readlines()
        start_line = 0
        for i, line in enumerate(lines):
            if line.startswith('Rank,Function,Score'):
                start_line = i
                break
        f.seek(0)
        for _ in range(start_line):
            next(f)
        reader = csv.DictReader(f)
        for row in reader:
            rank = row['Rank']
            function_field = row['Function']
            if rank in processed_ranks:
                print(f"跳过已处理的行: Rank {rank}")
                continue
            tasks.append((rank, function_field, endtime))
    fieldnames = ['Rank', 'Function', 'Count']
    if not os.path.exists(resultPath):
        with open(resultPath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        future_to_task = {executor.submit(process_single_function, task): task for task in tasks}
        for future in as_completed(future_to_task):
            try:
                result = future.result()
                print(result)
                with open(resultPath, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
            except Exception as e:
                print(f"处理任务时发生错误: {e}")
    total_time = time.time() - start_time
    print(total_time)
