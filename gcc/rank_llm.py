import json
import os
import re
import shutil
import stat
import subprocess
import time
from collections import Counter
from configparser import ConfigParser
from typing import List

from openai import OpenAI


def read_file_content(file_path):
    """读取文件内容，如果文件不存在则返回错误信息"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"无法读取文件 {file_path}: {str(e)}"


def extract_table_from_markdown(response_text):
    """从markdown响应中提取表格内容"""
    # 查找表格的开始和结束
    table_start = response_text.find('|')
    if table_start == -1:
        return None

    # 提取表格部分
    table_lines = []
    lines = response_text.split('\n')
    in_table = False

    for line in lines:
        if line.strip().startswith('|') and '---' not in line:
            in_table = True
            table_lines.append(line.strip())
        elif in_table and not line.strip().startswith('|'):
            break

    return '\n'.join(table_lines) if table_lines else None


def is_valid_number(s: str) -> bool:
    """检查字符串是否可以被安全地转换为浮点数。"""
    try:
        # 排除空字符串或纯空格
        if not s.strip():
            return False
        # 尝试转换，如果成功则为有效数字
        float(s)
        return True
    except ValueError:
        return False


def is_valid_filename_part(s: str) -> bool:
    """检查文件名部分是否主要包含英文、数字、点号和斜杠 (/)，用于路径。"""
    # 允许的字符包括：字母 (a-zA-Z), 数字 (0-9), 点号 (.), 斜杠 (/) 和下划线 (_)
    # 如果字符串中超过 80% 的字符是这些允许字符，我们认为它是有效的文件名。
    s_cleaned = s.strip()
    if not s_cleaned:
        return False

    # 统计允许字符的数量
    allowed_chars = re.sub(r'[a-zA-Z0-9./_-]', '', s_cleaned)

    # 允许一些非允许字符存在，如空格、括号等，但不能是纯数字或纯特殊符号
    # 只要它不是纯数字且包含路径或点号，就倾向于认为是文件名。
    if re.search(r'[a-zA-Z]', s_cleaned) or re.search(r'\.|/', s_cleaned):
        return True
    return False


def save_table_to_csv(table_text: str, output_path: str) -> bool:
    """
    将markdown表格转换为CSV格式并保存，并严格验证内容。
    第一列和第三列必须为数字，第二列必须为文件名（英文/点号/斜杠等）。
    """
    if not table_text:
        print("没有找到表格内容")
        return False

    try:
        lines = table_text.strip().split('\n')
        csv_lines: List[str] = []

        # 启发式判断表头：查找包含 'Rank' 和 'Score' 的行
        header_identified = False

        for line in lines:
            line = line.strip()

            # 1. 忽略空行
            if not line:
                continue

            # 2. 忽略 Markdown 分割线 (|---|---|---)
            if re.match(r'^\|[\s]*[-:=]+[\s]*\|', line):
                continue

            # 3. 忽略包含省略号 '...' 的行
            if '...' in line:
                continue

            # 4. 提取和清理单元格
            # 移除两侧的 '|'，并按 '|' 分割单元格
            cells = [cell.strip() for cell in line.strip('|').split('|')]

            # 确保至少有 3 列数据
            if len(cells) < 3:
                continue

            # 只取前三列
            selected_cells = cells[:3]
            clean_cells = [c.replace('\n', ' ').strip() for c in selected_cells]

            # 5. 表头识别和处理
            if not header_identified:
                # 检查是否包含表头关键词
                if all(keyword in clean_cells[i].lower() for i, keyword in enumerate(['rank', 'filename', 'score'])):
                    csv_lines.append(','.join(clean_cells))
                    header_identified = True
                    continue

            # 6. 数据行内容验证（只有在表头被识别后才开始验证）
            if header_identified:
                rank_str = clean_cells[0]
                filename_str = clean_cells[1]
                score_str = clean_cells[2]

                # a. 验证 Rank (第一列)
                is_rank_valid = is_valid_number(rank_str)

                # b. 验证 Score (第三列)
                is_score_valid = is_valid_number(score_str)

                # c. 验证 Filename (第二列)
                is_filename_valid = is_valid_filename_part(filename_str)

                # 只有当 Rank 和 Score 都是有效数字，且 Filename 看起来像有效路径时，才保留该行
                if is_rank_valid and is_score_valid and is_filename_valid:
                    csv_lines.append(','.join(clean_cells))
                else:
                    # 打印被丢弃的行，便于调试
                    # print(f"警告：丢弃无效数据行 - Rank({is_rank_valid}), File({is_filename_valid}), Score({is_score_valid}): {line}")
                    pass

        # 检查是否有实际数据被提取
        if len(csv_lines) < 2:  # 至少需要表头和一行数据
            print(f"提取到的有效数据行数不足: {len(csv_lines)} 行 (少于1行数据)")
            return False

        # 保存CSV文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(csv_lines))

        print(f"表格已保存为: {output_path}")
        return True
    except Exception as e:
        print(f"保存CSV文件失败: {str(e)}")
        return False


def run_executable(exe_path):
    """运行可执行文件（使用绝对路径，修复找不到文件问题）"""
    exe_abs_path = os.path.abspath(exe_path)  # 转为绝对路径
    try:
        # 先验证文件是否真的存在
        if not os.path.exists(exe_abs_path):
            return {"success": False, "exit_code": -4, "output": "", "error": f"文件不存在（绝对路径）: {exe_abs_path}"}

        result = subprocess.run(
            [exe_abs_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            timeout=5  # 超时控制
        )
        return {
            "success": True,
            "exit_code": result.returncode,
            "output": result.stdout.strip(),
            "error": ""
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "exit_code": -1, "output": "", "error": f"运行超时（绝对路径）: {exe_abs_path}"}
    except Exception as e:
        return {"success": False, "exit_code": -2, "output": "",
                "error": f"运行异常（绝对路径）: {str(e)} | 路径: {exe_abs_path}"}


def fix_executable_permission(exe_path):
    """修复可执行文件权限"""
    exe_abs_path = os.path.abspath(exe_path)
    if os.path.exists(exe_abs_path):
        os.chmod(exe_abs_path, stat.S_IRWXU)
        return True, exe_abs_path
    return False, exe_abs_path


def get_detailed_compile_logs_gcc(compiler, src, opt, output_prefix):
    """
    GCC版本的详细编译日志生成（替换原Clang的LLVM IR为GCC的GIMPLE/RTL中间表示）
    """
    logs = {
        "preprocess": "", "ast": "", "gimple": "", "rtl": "",
        "asm": "", "opt_passes": "", "backend": ""
    }

    # 1. 预处理（GCC与Clang参数兼容，保留-E，强制C语言预处理）
    try:
        pre_cmd = [compiler, src, opt, "-E", "-x", "c", "-o", f"{output_prefix}.i"]
        pre_result = subprocess.run(pre_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8')
        # 写入预处理文件
        with open(f"{output_prefix}.i", 'w', encoding='utf-8') as f:
            f.write(pre_result.stdout)
        line_count = len(pre_result.stdout.splitlines()) if pre_result.stdout else 0
        logs[
            "preprocess"] = f"预处理命令：{' '.join(pre_cmd)}\n输出文件：{output_prefix}.i\n预处理后代码行数：{line_count}\n"
        if line_count == 0:
            logs["preprocess"] += "警告：预处理行数为0，可能是测试文件为空或命令异常\n"
    except Exception as e:
        logs["preprocess"] = f"预处理失败：{str(e)}\n"

    # 2. AST分析（GCC使用-fdump-tree-ast生成AST，替换Clang的-ast-dump）
    try:
        ast_dump_file = f"{output_prefix}.ast"
        ast_cmd = [compiler, src, opt, "-fdump-tree-ast", "-fsyntax-only", "-o", ast_dump_file]
        ast_result = subprocess.run(ast_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8')
        # 解析GCC的AST dump文件（提取函数/变量声明数）
        ast_content = read_file_content(ast_dump_file) if os.path.exists(ast_dump_file) else ""
        funcs = len(re.findall(r'function_decl', ast_content, re.IGNORECASE))
        vars = len(re.findall(r'var_decl', ast_content, re.IGNORECASE))
        exprs = len(re.findall(r'expr', ast_content, re.IGNORECASE))
        logs["ast"] = f"AST生成命令：{' '.join(ast_cmd)}\nAST输出文件：{ast_dump_file}\n函数声明数：{funcs} | 变量声明数：{vars} | 表达式数：{exprs}\n"
        logs["ast"] += f"AST输出预览（前5行）：\n{chr(10).join(ast_content.splitlines()[:5])}...\n" if ast_content else "AST文件为空\n"
    except Exception as e:
        logs["ast"] = f"AST生成异常：{str(e)}\n"

    # 3. GIMPLE中间表示（GCC核心中间表示，替换Clang的未优化IR）
    try:
        gimple_dump_file = f"{output_prefix}.gimple"
        gimple_cmd = [compiler, src, opt, "-O0", "-fdump-tree-gimple", "-fsyntax-only", "-o", gimple_dump_file]
        subprocess.run(gimple_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8')
        gimple_content = read_file_content(gimple_dump_file) if os.path.exists(gimple_dump_file) else ""
        gimple_lines = len([l for l in gimple_content.splitlines() if l.strip()]) if gimple_content else 0
        logs["gimple"] = f"GIMPLE生成命令：{' '.join(gimple_cmd)}\nGIMPLE输出文件：{gimple_dump_file}\n有效GIMPLE行数：{gimple_lines}\n"
    except Exception as e:
        logs["gimple"] = f"GIMPLE生成失败：{str(e)}\n"

    # 4. RTL中间表示（GCC后端中间表示，替换Clang的优化IR）
    try:
        rtl_dump_file = f"{output_prefix}.rtl"
        rtl_cmd = [compiler, src, opt, "-fdump-rtl-all", "-fsyntax-only", "-o", rtl_dump_file]
        subprocess.run(rtl_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8')
        rtl_content = read_file_content(rtl_dump_file) if os.path.exists(rtl_dump_file) else ""
        rtl_lines = len([l for l in rtl_content.splitlines() if l.strip()]) if rtl_content else 0
        logs["rtl"] = f"RTL生成命令：{' '.join(rtl_cmd)}\nRTL输出文件：{rtl_dump_file}\n有效RTL行数：{rtl_lines}\n"
    except Exception as e:
        logs["rtl"] = f"RTL生成失败：{str(e)}\n"

    # 5. 优化Pass日志（GCC使用-fdump-passes查看优化流程，替换Clang的-debug-pass=Structure）
    try:
        pass_dump_file = f"{output_prefix}.passes"
        pass_cmd = [compiler, src, opt, "-fdump-passes", "-fsyntax-only", "-o", pass_dump_file]
        pass_result = subprocess.run(pass_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding='utf-8')
        # 提取优化Pass列表（过滤关键行）
        pass_lines = [line.strip() for line in pass_result.stdout.splitlines() if "pass" in line.lower() and line.strip()]
        logs[
            "opt_passes"] = f"优化Pass命令：{' '.join(pass_cmd)}\nPass输出文件：{pass_dump_file}\n优化Pass总数：{len(pass_lines)}\n关键Pass列表（前20行）：\n{chr(10).join(pass_lines[:20])}...\n" if pass_lines else "未提取到优化Pass\n"
    except Exception as e:
        logs["opt_passes"] = f"优化Pass日志获取失败：{str(e)}\n"

    # 6. 汇编代码（GCC与Clang参数兼容，保留-fverbose-asm）
    try:
        asm_cmd = [compiler, src, opt, "-S", "-fverbose-asm", "-o", f"{output_prefix}.s"]
        subprocess.run(asm_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8')
        with open(f"{output_prefix}.s", 'r', encoding='utf-8') as f:
            asm_lines = [line for line in f.readlines() if not line.startswith(('#', '.')) and line.strip()]
        logs["asm"] = f"汇编生成命令：{' '.join(asm_cmd)}\n输出文件：{output_prefix}.s\n有效汇编指令数：{len(asm_lines)}\n"
    except Exception as e:
        logs["asm"] = f"汇编生成失败：{str(e)}\n"

    # 7. 后端编译日志（GCC的-v参数输出后端流程，与Clang兼容）
    try:
        backend_cmd = [compiler, src, opt, "-c", "-o", f"{output_prefix}.o", "-v"]
        backend_result = subprocess.run(backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8')
        backend_info = [line for line in backend_result.stdout.splitlines() if
                        "as" in line or "target" in line or "gcc version" in line]
        logs[
            "backend"] = f"后端编译命令：{' '.join(backend_cmd)}\n输出目标文件：{output_prefix}.o\n后端关键信息：\n" + "\n".join(
            backend_info) + "\n"
    except Exception as e:
        logs["backend"] = f"后端编译失败：{str(e)}\n"

    return logs


def fileRank_llm(bugid, rev, configFile, rightOption, wrongOption, k):
    cfg = ConfigParser()
    cfg.read(configFile)
    abstractDir = cfg.get('gcc-locations', 'abstractDir')
    infoBasePath = cfg.get('gcc-locations', 'infodir')
    compilerBasePath = cfg.get('gcc-locations', 'compilersdir')
    basePassDir = cfg.get('gcc-locations', 'passdir')
    logBaseDir = cfg.get('gcc-locations', 'logDir')
    structureFile = cfg.get('gcc-locations', 'structureFile')
    key = cfg.get('gpt-gemini','api')
    url = cfg.get('gpt-gemini','url')
    filemap = dict()
    methodcov = infoBasePath + bugid + '/failcov/method_info.txt'
    if not os.path.exists(logBaseDir + bugid):
        os.mkdir(logBaseDir + bugid)
    if not os.path.exists(os.path.join(logBaseDir, bugid, 'fail.c')):
        source_file = os.path.join(infoBasePath, bugid, 'fail.c')
        destination_file = os.path.join(logBaseDir, bugid, 'fail.c')
        try:
            shutil.copyfile(source_file, destination_file)
            print(f"成功复制文件: {source_file} -> {destination_file}")
        except FileNotFoundError:
            print(f"源文件不存在: {source_file}")
        except Exception as e:
            print(f"复制文件失败: {str(e)}")
    os.chdir(infoBasePath + bugid)
    with open(methodcov, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()  # 移除首尾换行/空格，避免拆分出空字符串
            if not line:  # 跳过空行
                continue
            items = line.split(',')
            # 格式要求至少4列：gcda文件,函数名,百分比,执行行数描述 → 不足则跳过
            if len(items) < 4:
                continue

            # 步骤1：提取文件名（保留原有逻辑）
            filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]

            # 步骤2：提取覆盖率百分比（第三列，如100.00）
            try:
                coverage_percent = float(items[2].strip())  # 转浮点数
            except (ValueError, IndexError):
                # 第三列无法转数字则跳过
                continue

            # 步骤3：从第四列提取总行数（正则匹配“共 X 行”中的X）
            total_lines_str = re.search(r'共 (\d+) 行', items[3])
            if not total_lines_str:
                # 未匹配到总行数则跳过
                continue
            try:
                total_lines = int(total_lines_str.group(1))  # 转整数
            except ValueError:
                continue

            # 步骤4：计算「总行数 × 覆盖率百分比/100」的结果（执行行数）
            # 如需整数则用 round() 或 int() 转换，如：exec_lines = int(total_lines * coverage_percent / 100)
            exec_lines = total_lines * (coverage_percent / 100)

            # 步骤5：累加计算结果到filemap（替换原有调用次数累加）
            filemap[filename] = filemap.get(filename, 0) + exec_lines

    # 收集结果并写入 abstract.txt
    result_lines = []
    for filename, call_count in filemap.items():
        # 1. 获取无后缀的文件名基名（原有逻辑）
        temp = os.path.basename(filename)
        # 2. 构造.cc.json和.h.json的路径（核心修改）
        cc_json_path = os.path.join(abstractDir, f"{temp}.cc.json")
        h_json_path = os.path.join(abstractDir, f"{temp}.h.json")
        # 3. 初始化两个文件的摘要信息（默认标注为无）
        cc_summary = "无.cc文件摘要信息"
        h_summary = "无.h文件摘要信息"
        # 4. 单独读取.cc.json并处理异常
        if os.path.exists(cc_json_path):
            try:
                with open(cc_json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    cc_summary = data.get('summary', '无.cc文件摘要信息')
            except json.JSONDecodeError:
                cc_summary = f".cc.json格式错误"
            except Exception as e:
                cc_summary = f".cc.json读取错误 - {str(e)}"
        # 5. 单独读取.h.json并处理异常
        if os.path.exists(h_json_path):
            try:
                with open(h_json_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    h_summary = data.get('summary', '无.h文件摘要信息')
            except json.JSONDecodeError:
                h_summary = f".h.json格式错误"
            except Exception as e:
                h_summary = f".h.json读取错误 - {str(e)}"
        # 6. 合并摘要信息（可自定义分隔符，如分号、换行等）
        merged_summary = f"[CC摘要]：{cc_summary} | [H摘要]：{h_summary}"
        # 7. 写入结果列表（保留原有格式，替换为合并后的摘要）
        result_lines.append(f"{filename}，调用次数:{call_count}：{merged_summary}")

    with open(logBaseDir + bugid + '/abstract.txt', 'w', encoding='utf-8') as out_file:
        out_file.write('\n'.join(result_lines))

    # ========== GCC编译器路径构造（替换原Clang路径） ==========
    compiler = os.path.join(compilerBasePath, rev, f'{rev}-build', 'bin', 'gcc')
    fail = os.path.join(infoBasePath, bugid, 'fail.c')

    def compile_deep_analysis(log_file):
        compile_configs = [
            {"name": rightOption, "opt": rightOption, "prefix": f"{rightOption}_deep"},
            {"name": wrongOption, "opt": wrongOption, "prefix": f"{wrongOption}_deep"}
        ]

        # 日志头部
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"深度编译分析日志（GCC版本） | BugID: {bugid}\n")
            f.write(f"测试文件: {fail}\n")
            f.write(f"编译架构: x86-64 | GCC版本: {rev} | 调试选项: -g \n")
            f.write("=" * 100 + "\n\n")

        for cfg in compile_configs:
            exe_path = logBaseDir + bugid + '/' + f"output_{cfg['name'].lower()}"
            # GCC编译命令（移除Clang独有参数，保留GCC兼容参数）
            full_cmd = [
                compiler, fail, cfg["opt"], "-march=x86-64", "-g",
                "-ftime-report", "-v",
                "-o", exe_path
            ]

            start_time = time.time()
            try:
                # 执行编译
                compile_result = subprocess.run(
                    full_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding='utf-8'
                )
                duration = round(time.time() - start_time, 2)
                compile_success = compile_result.returncode == 0
                exe_exists, exe_abs_path = fix_executable_permission(exe_path) if compile_success else (False, "")
                file_size = os.path.getsize(exe_abs_path) if (compile_success and exe_exists) else 0
                # 调用GCC版本的编译日志生成函数
                stage_logs = get_detailed_compile_logs_gcc(compiler, fail, cfg["opt"], cfg["prefix"])
                run_result = run_executable(exe_abs_path) if (compile_success and exe_exists) else {"success": False,
                                                                                                    "exit_code": -3,
                                                                                                    "output": "",
                                                                                                    "error": "编译失败或可执行文件不存在"}
                # GCC的耗时报告提取（-ftime-report的输出）
                time_report = [line for line in compile_result.stdout.splitlines() if
                               "Time Report" in line or "user" in line or "sys" in line or "wall" in line]
                time_report_str = "\n".join(time_report) if time_report else "无详细优化耗时日志"

                # 写入当前配置日志
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"【{cfg['name']} 深度分析（GCC）】\n")
                    f.write("=" * 100 + "\n")
                    f.write(f"1. 基础信息\n")
                    f.write(f"   编译命令：{' '.join(full_cmd)}\n")
                    f.write(
                        f"   编译结果：{'成功' if compile_success else '失败'} | 返回码：{compile_result.returncode} | 总耗时：{duration}s\n")
                    f.write(
                        f"   产物信息：存在: {'是' if exe_exists else '否'} | 绝对路径: {exe_abs_path} | 大小：{file_size}B\n")
                    f.write(
                        f"   运行结果：{'正常' if run_result['success'] else '异常'} | 退出码：{run_result['exit_code']}\n")
                    f.write(f"   运行输出：{run_result['output'] or '无'}\n")
                    f.write(f"   运行错误：{run_result['error']}\n\n")

                    # 写入GCC各阶段日志（替换原Clang的IR相关）
                    f.write(f"2. 预处理阶段\n")
                    f.write(f"   {stage_logs['preprocess']}\n")
                    f.write(f"3. AST分析阶段\n")
                    f.write(f"   {stage_logs['ast']}\n")
                    f.write(f"4. GCC中间表示阶段（GIMPLE/RTL）\n")
                    f.write(f"   {stage_logs['gimple']}\n")
                    f.write(f"   {stage_logs['rtl']}\n")
                    f.write(f"5. 优化Pass详情\n")
                    f.write(f"   {stage_logs['opt_passes']}\n\n")
                    f.write(f"6. 汇编生成阶段\n")
                    f.write(f"   {stage_logs['asm']}\n")
                    f.write(f"7. 后端编译阶段\n")
                    f.write(f"   {stage_logs['backend']}\n")
                    f.write(f"8. 编译耗时详情\n")
                    f.write(f"   {time_report_str}\n")
                    f.write("=" * 100 + "\n\n")

            except Exception as e:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(f"【{cfg['name']} 分析异常（GCC）】\n")
                    f.write(f"   异常信息：{str(e)}\n")
                    f.write("=" * 100 + "\n\n")

        # 跨配置差异对比（适配GCC）
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("【跨配置核心差异对比（GCC）】\n")
            f.write("=" * 100 + "\n")

            # 1. 优化Pass差异（GCC版本）
            try:
                o2_pass_lines = []
                o3_pass_lines = []
                o2_pass_path = f"{compile_configs[0]['prefix']}.passes"
                o3_pass_path = f"{compile_configs[1]['prefix']}.passes"

                # 重新生成两个优化级别的Pass日志
                o2_pass_cmd = [compiler, fail, rightOption, "-fdump-passes", "-fsyntax-only", "-o", o2_pass_path]
                o2_pass_result = subprocess.run(o2_pass_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                               encoding='utf-8')
                o2_pass_lines = [line.strip() for line in o2_pass_result.stdout.splitlines() if "pass" in line.lower()]

                o3_pass_cmd = [compiler, fail, wrongOption, "-fdump-passes", "-fsyntax-only", "-o", o3_pass_path]
                o3_pass_result = subprocess.run(o3_pass_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                               encoding='utf-8')
                o3_pass_lines = [line.strip() for line in o3_pass_result.stdout.splitlines() if "pass" in line.lower()]

                # 提取独有Pass
                o2_only_passes = [p for p in o2_pass_lines if p not in o3_pass_lines]
                o3_only_passes = [p for p in o3_pass_lines if p not in o2_pass_lines]

                f.write(f"1. 优化Pass差异\n")
                f.write(f"   {rightOption}总Pass数：{len(o2_pass_lines)} | {wrongOption}总Pass数：{len(o3_pass_lines)}\n")
                f.write(
                    f"   {rightOption}独有Pass（{len(o2_only_passes)}个）：\n     {chr(10).join(o2_only_passes[:20])}...\n" if o2_only_passes else "   无\n")
                f.write(
                    f"   {wrongOption}独有Pass（{len(o3_only_passes)}个）：\n     {chr(10).join(o3_only_passes[:20])}...\n" if o3_only_passes else "   无\n")
            except Exception as e:
                f.write(f"1. 优化Pass差异对比失败：{str(e)}\n")

            # 2. 汇编指令差异（GCC与Clang生成的汇编格式略有不同，但解析逻辑兼容）
            try:
                o2_asm = []
                o3_asm = []
                o2_s_path = f"{compile_configs[0]['prefix']}.s"
                o3_s_path = f"{compile_configs[1]['prefix']}.s"

                if os.path.exists(o2_s_path):
                    with open(o2_s_path, 'r', encoding='utf-8') as asm_file:
                        o2_asm = [line.strip().split()[0] for line in asm_file if
                                  not line.startswith(('#', '.')) and line.strip()]
                if os.path.exists(o3_s_path):
                    with open(o3_s_path, 'r', encoding='utf-8') as asm_file:
                        o3_asm = [line.strip().split()[0] for line in asm_file if
                                  not line.startswith(('#', '.')) and line.strip()]

                o2_inst_cnt = Counter(o2_asm)
                o3_inst_cnt = Counter(o3_asm)
                # 找出指令计数差异
                diff_inst = []
                all_inst = set(o2_inst_cnt.keys()).union(set(o3_inst_cnt.keys()))
                for inst in all_inst:
                    o2_cnt = o2_inst_cnt.get(inst, 0)
                    o3_cnt = o3_inst_cnt.get(inst, 0)
                    if o2_cnt != o3_cnt:
                        diff_inst.append(f"{inst}: {rightOption}={o2_cnt} | {wrongOption}={o3_cnt}")

                f.write(f"\n2. 汇编指令差异\n")
                f.write(f"   {rightOption}有效指令数：{len(o2_asm)} | {wrongOption}有效指令数：{len(o3_asm)}\n")
                f.write(f"   指令计数差异：\n     {chr(10).join(diff_inst) if diff_inst else '无'}\n")
                f.write(f"   {rightOption}指令分布：{dict(o2_inst_cnt) if o2_inst_cnt else '无'}\n")
                f.write(f"   {wrongOption}指令分布：{dict(o3_inst_cnt) if o3_inst_cnt else '无'}\n")
            except Exception as e:
                f.write(f"\n2. 汇编差异对比失败：{str(e)}\n")

    compile_deep_analysis(logBaseDir + bugid + "/deep_compile_analysis_gcc.log")
    print(f"GCC版本编译分析日志已生成：{logBaseDir}{bugid}/deep_compile_analysis_gcc.log")
    client = OpenAI(
        api_key=key,
        base_url=url,
        timeout=3600
    )
    # 读取三个文件的内容
    abstract_file = logBaseDir + bugid + '/abstract.txt'
    deep_compile_file = logBaseDir + bugid + '/deep_compile_analysis_gcc.log'  # 改为GCC日志文件
    fail_c_file = logBaseDir + bugid + '/fail.c'
    pass_code_dir = basePassDir + bugid + '/passing_cases/'
    fail_code_dir = basePassDir + bugid + '/failing_cases/'
    pass_code_list = []
    fail_code_list = []
    abstract_content = ''
    deep_compile_content = ''
    fail_c_content = ''
    structure_content = ''
    print("正在读取文件内容...")
    if os.path.exists(abstract_file):
        abstract_content = read_file_content(abstract_file)
    if os.path.exists(deep_compile_file):
        deep_compile_content = read_file_content(deep_compile_file)
    if os.path.exists(fail_c_file):
        fail_c_content = read_file_content(fail_c_file)
    if os.path.exists(structureFile):
        structure_content = read_file_content(structureFile)
    for i in range(6):
        if os.path.exists(pass_code_dir + f"pass_{i:04d}.c"):
            pass_code_list.append(read_file_content(pass_code_dir + f"pass_{i:04d}.c"))
    for i in range(6):
        if os.path.exists(fail_code_dir + f"fail_{i:04d}.c"):
            fail_code_list.append(read_file_content(fail_code_dir + f"fail_{i:04d}.c"))

    # 构建提示词（标注GCC版本）
    system_prompt = """你是一个编译器优化问题分析专家，需要基于GCC编译器的多源信息对可疑文件进行重排序分析。"""

    user_prompt = f"""
    你需要基于GCC编译器的多源信息重排序可疑文件，你要根据文件执行次数，相关文件功能摘要文档，测试用例代码和GCC编译输出信息，结合测试用例差异、失败特征、文件执行次数与文件功能描述分析相关性，重点关注GCC不同优化水平（O0/O1/O2/O3等）的特点，为每个文件分配0-10分并按分数降序排列。
    请按以下格式输出：
    1. 首先进行详细的思维分析过程，分析每个文件与bug的相关性（需结合GCC编译特性）
    2. 然后用markdown表格格式给出最终排序结果，表格包含三列：Rank, Filename, Score，排名尽可能完整，包括所有文件
    以下是几个关键文件的内容：
    {'=== abstract.txt (文件执行次数和功能摘要) ===' + abstract_content if len(abstract_content) != 0 else ''}
    {'=== deep_compile_analysis_gcc.log (GCC编译分析日志) ===' + deep_compile_content if len(deep_compile_content) != 0 else ''}
    {'=== fail.c (测试用例代码) ===' + fail_c_content if len(fail_c_content) != 0 else ''}
    {'=== 根据fail.c变异的成功测试用例 === ' + chr(10).join(pass_code_list) if len(pass_code_list) != 0 else ''}
    {'=== 根据fail.c变异的失败测试用例 === ' + chr(10).join(fail_code_list) if len(fail_code_list) != 0 else ''}
    请基于以上GCC相关信息进行深度分析，并提供最终的文件排序结果。
    """

    print(user_prompt)
    try:
        print("正在发送请求到LLM API...")
        response = client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=1,
            max_tokens=50000
        )

        # 获取响应内容
        response_text = response.choices[0].message.content
        print("收到API响应")

        # 保存完整响应
        full_response_path = logBaseDir + bugid + f'/llm_full_response_{k}.md'
        with open(full_response_path, 'w', encoding='utf-8') as f:
            f.write(response_text)
        print(f"GCC版本完整响应已保存: {full_response_path}")

        # 提取表格内容
        table_content = extract_table_from_markdown(response_text)

        if table_content:
            print("找到表格内容，正在转换为CSV格式...")
            # 保存表格为CSV
            result_csv_path = logBaseDir + bugid + f'/result_llm_{k}.csv'
            if save_table_to_csv(table_content, result_csv_path):
                print(f"✅ 成功生成GCC版本结果文件: {result_csv_path}")
            else:
                print("❌ 表格转换失败")
        else:
            print("❌ 在响应中未找到表格内容")
            # 保存原始响应以供调试
            debug_path = logBaseDir + bugid + f'/llm_response_debug_{k}.txt'
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(response_text)
            print(f"GCC版本调试信息已保存: {debug_path}")

    except Exception as e:
        print(f"❌ API调用失败: {str(e)}")

    print("GCC版本LLM分析流程完成")