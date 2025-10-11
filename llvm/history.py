import csv
import os.path
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
import time

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

def process_single_function(args):
    rank, function_field, endtime = args
    if ';' in function_field:
        file_path, function_signature = function_field.split(';', 1)
    else:
        print(f"警告: 无法解析函数字段: {function_field}")
        return {
            'Rank': rank,
            'Function': function_field,
            'Count': -1
        }
    function_name = extract_function_name(function_signature)
    try:
        # 获取包含diff的git log
        cmd = [
            'git', 'log',
            '--since', starttime,
            '--until', endtime,
            '--follow',
            '--pretty=format:%H',
            '-p',  # 显示diff
            '--', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # 手动解析输出，查找包含函数名的diff
        commit_hashes = set()
        lines = result.stdout.split('\n')
        current_commit = None
        for line in lines:
            if (len(line) == 40 and
                    all(c in '0123456789abcdef' for c in line.lower()) and
                    not line.startswith(('+', '-', '@', 'diff', 'index'))):
                current_commit = line
            elif line.startswith('+') or line.startswith('-'):
                if function_name in line and f"{function_name}(" in line:
                    line_content = line[1:].strip()
                    if not (line_content.startswith('//') or line_content.startswith('/*') or line_content.startswith(
                            '*')):
                        if current_commit:
                            commit_hashes.add(current_commit)
        count = len(commit_hashes)

    except subprocess.CalledProcessError as e:
        print(f"执行git命令出错: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return {
            'Rank': rank,
            'Function': function_field,
            'Count': 0
        }

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
                with open(resultPath, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(result)
            except Exception as e:
                print(f"处理任务时发生错误: {e}")
    total_time = time.time() - start_time
    print(total_time)