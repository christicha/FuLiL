import os
import stat
import json
import re
import time
import subprocess
from collections import Counter
from configparser import ConfigParser

# 读取配置文件
config = ConfigParser()
config.read('../config/config.ini', encoding='utf-8')
abstractDir = config.get('llvm-locations', 'abstractDir')
infoBasePath = config.get('llvm-locations', 'infodir')
compilerBasePath = config.get('llvm-locations', 'compilersdir')

# 基础参数
bugid = '15920'
revision = 'r181189'

# ---------------------- 第一步：处理 method_info.txt 生成 abstract.txt ----------------------
filemap = dict()
methodcov = infoBasePath + bugid + '/failcov/method_info.txt'
with open(methodcov, 'r') as f:
    lines = f.readlines()
    for line in lines:
        items = line.split(',')
        if len(items) < 2:  # 跳过格式异常的行
            continue
        filename = re.sub(r'/CMakeFiles/[^/]+.dir', '', items[0]).split('.gcda')[0]
        # 累加最后一列的数值（调用次数）
        try:
            count = int(items[-1].strip())
            filemap[filename] = filemap.get(filename, 0) + count
        except ValueError:
            continue  # 跳过无法转换为整数的行

# 收集结果并写入 abstract.txt
result_lines = []
for filename, call_count in filemap.items():
    temp = os.path.basename(filename)
    json_path = os.path.join(abstractDir, f"{temp}.json")
    print(json_path)
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                summary = data.get('summary', '无摘要信息')
                result_lines.append(f"{filename}，调用次数:{call_count}：{summary}")
        except json.JSONDecodeError:
            print(f"{filename}，{call_count}：JSON格式错误")
        except Exception as e:
            print(f"{filename}，{call_count}：读取错误 - {str(e)}")
    else:
        print(f"{filename}，{call_count}：对应JSON文件不存在")

with open('abstract.txt', 'w', encoding='utf-8') as out_file:
    out_file.write('\n'.join(result_lines))

# ---------------------- 编译相关基础配置 ----------------------
compiler = os.path.join(compilerBasePath, revision, f'{revision}-build', 'bin', 'clang')
fail = os.path.join(infoBasePath, bugid, 'fail.c')
rightOption = '-O2'
wrongOption = '-O3'

# 前置检查
if not os.path.exists(compiler):
    raise FileNotFoundError(f"编译器不存在: {compiler}")
if not os.path.exists(fail):
    raise FileNotFoundError(f"测试文件不存在: {fail}")


# ---------------------- 通用工具函数 ----------------------
def get_llvm_ir(compiler, src, opt, output_ir):
    """生成LLVM中间代码（IR）"""
    cmd = [compiler, src, opt, '-S', '-emit-llvm', '-o', output_ir]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
        return True
    except:
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


# ---------------------- 函数1：生成简洁缺陷定位日志 ----------------------
def compile_and_debug(log_file):
    compile_configs = [
        {"name": "O2", "opt": rightOption, "ir": "o2.ll", "asm": "o2.s", "exe": "output_o2"},
        {"name": "O3", "opt": wrongOption, "ir": "o3.ll", "asm": "o3.s", "exe": "output_o3"}
    ]

    # 日志头部
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"缺陷定位日志 | BugID: {bugid} \n")
        f.write(f"测试文件: {fail}\n")
        f.write("-" * 80 + "\n\n")

    # 编译并收集关键信息
    for cfg in compile_configs:
        cmd = [
            compiler, fail,
            cfg["opt"], "-march=x86-64", "-g",
            "-ftime-report",
            "-o", cfg["exe"]
        ]
        start = time.time()
        try:
            compile_result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8'
            )
            duration = round(time.time() - start, 2)
            success = compile_result.returncode == 0
            exe_exists, exe_abs_path = fix_executable_permission(cfg["exe"]) if success else (False, "")
            file_size = os.path.getsize(exe_abs_path) if (success and exe_exists) else 0
            ir_success = get_llvm_ir(compiler, fail, cfg["opt"], cfg["ir"]) if success else False
            run_info = run_executable(exe_abs_path) if (success and exe_exists) else {"success": False, "exit_code": -3,
                                                                                      "output": "",
                                                                                      "error": "编译失败，未生成可执行文件"}

            # 写入日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"【{cfg['name']} 配置】\n")
                f.write(f"  编译命令: {' '.join(cmd)}\n")
                f.write(
                    f"  编译结果: {'成功' if success else '失败'} | 返回码: {compile_result.returncode} | 耗时: {duration}s\n")
                f.write(
                    f"  产物信息: 存在: {'是' if exe_exists else '否'} | 绝对路径: {exe_abs_path} | 大小: {file_size}B\n")
                f.write(
                    f"  运行结果: {'正常退出' if run_info['success'] else '异常'} | 退出码: {run_info['exit_code']}\n")
                f.write(f"  运行输出: {run_info['output'] if run_info['output'] else '无'}\n")
                f.write(f"  运行错误: {run_info['error']}\n")
                f.write(
                    f"  优化日志: {compile_result.stdout.split('=== Time Report ===')[-1].strip() if '=== Time Report ===' in compile_result.stdout else '无'}\n")
                f.write("-" * 80 + "\n\n")

        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"【{cfg['name']} 配置】\n")
                f.write(f"  编译异常: {str(e)}\n")
                f.write("-" * 80 + "\n\n")

    # IR差异对比（适配旧版llvm-diff）
    try:
        if os.path.exists(compile_configs[0]["ir"]) and os.path.exists(compile_configs[1]["ir"]):
            ir_diff = subprocess.run(
                ["llvm-diff", compile_configs[0]["ir"], compile_configs[1]["ir"]],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8'
            )
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write("【LLVM IR 差异对比】\n")
                f.write(
                    f"  差异内容: {ir_diff.stdout.strip() if ir_diff.stdout else '无差异（但O2/O3优化pass不同，需看深度日志）'}\n")
                f.write("-" * 80 + "\n\n")
    except:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("【LLVM IR 差异对比】\n  对比失败（可能缺少llvm-diff工具或版本不兼容）\n")
            f.write("-" * 80 + "\n\n")


# ---------------------- 函数2：生成深度编译分析日志（适配clang 3.3） ----------------------
def get_detailed_compile_logs(compiler, src, opt, output_prefix):
    """拆解编译阶段，适配clang 3.3（移除不支持的参数）"""
    logs = {
        "preprocess": "", "ast": "", "ir_raw": "", "ir_opt": "",
        "asm": "", "opt_passes": "", "backend": ""
    }

    # 1. 预处理（修复行数为0的问题：显式指定输入类型，确保正确预处理）
    try:
        pre_cmd = [compiler, src, opt, "-E", "-x", "c", "-o", f"{output_prefix}.i"]  # -x c：强制按C语言预处理
        pre_result = subprocess.run(pre_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8')
        # 写入预处理文件（避免stdout截断）
        with open(f"{output_prefix}.i", 'w', encoding='utf-8') as f:
            f.write(pre_result.stdout)
        line_count = len(pre_result.stdout.splitlines()) if pre_result.stdout else 0
        logs[
            "preprocess"] = f"预处理命令：{' '.join(pre_cmd)}\n输出文件：{output_prefix}.i\n预处理后代码行数：{line_count}\n"
        # 补充预处理是否成功的判断
        if line_count == 0:
            logs["preprocess"] += "警告：预处理行数为0，可能是测试文件为空或命令异常\n"
    except Exception as e:
        logs["preprocess"] = f"预处理失败：{str(e)}\n"

    # 2. AST分析（适配clang 3.3：用-ast-dump而非json，提取关键信息）
    try:
        ast_cmd = [compiler, src, opt, "-Xclang", "-ast-dump", "-fsyntax-only"]  # 旧版无json选项
        ast_result = subprocess.run(ast_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8')
        # 从AST输出中提取函数、变量、表达式数量（旧版文本格式解析）
        funcs = len(re.findall(r'FunctionDecl', ast_result.stdout))
        vars = len(re.findall(r'VarDecl', ast_result.stdout))
        exprs = len(re.findall(r'Expr', ast_result.stdout))
        logs["ast"] = f"AST命令：{' '.join(ast_cmd)}\n函数声明数：{funcs} | 变量声明数：{vars} | 表达式数：{exprs}\n"
        logs["ast"] += f"AST输出预览（前5行）：\n{chr(10).join(ast_result.stdout.splitlines()[:5])}...\n"
    except Exception as e:
        logs["ast"] = f"AST生成异常：{str(e)}\n"

    # 3. 未优化IR（保持不变）
    try:
        ir_raw_cmd = [compiler, src, opt, "-S", "-emit-llvm", "-O0", "-o", f"{output_prefix}_raw.ll"]
        subprocess.run(ir_raw_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8')
        logs["ir_raw"] = f"未优化IR命令：{' '.join(ir_raw_cmd)}\n输出文件：{output_prefix}_raw.ll\n"
    except Exception as e:
        logs["ir_raw"] = f"未优化IR生成失败：{str(e)}\n"

    # 4. 优化后IR + 优化pass详情（完整输出，不截断）
    try:
        ir_opt_cmd = [compiler, src, opt, "-S", "-emit-llvm", "-o", f"{output_prefix}_opt.ll", "-debug-pass=Structure"]
        opt_result = subprocess.run(ir_opt_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    encoding='utf-8')
        # 提取所有优化pass（过滤冗余，保留Running pass行）
        pass_lines = [line.strip() for line in opt_result.stdout.splitlines() if "Running pass" in line]
        logs[
            "opt_passes"] = f"优化IR命令：{' '.join(ir_opt_cmd)}\n输出文件：{output_prefix}_opt.ll\n优化pass执行顺序（共{len(pass_lines)}个）：\n" + "\n".join(
            pass_lines)
        logs["ir_opt"] = f"优化IR生成成功\n"
    except Exception as e:
        logs["ir_opt"] = f"优化IR生成失败：{str(e)}\n"
        logs["opt_passes"] = f"优化pass日志获取失败：{str(e)}\n"

    # 5. 汇编代码（保持不变，补充指令差异提示）
    try:
        asm_cmd = [compiler, src, opt, "-S", "-fverbose-asm", "-o", f"{output_prefix}.s"]
        subprocess.run(asm_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       encoding='utf-8')
        with open(f"{output_prefix}.s", 'r', encoding='utf-8') as f:
            asm_lines = [line for line in f.readlines() if not line.startswith(('#', '.')) and line.strip()]
        logs["asm"] = f"汇编命令：{' '.join(asm_cmd)}\n输出文件：{output_prefix}.s\n有效指令数：{len(asm_lines)}\n"
    except Exception as e:
        logs["asm"] = f"汇编生成失败：{str(e)}\n"

    # 6. 后端生成日志（保持不变）
    try:
        backend_cmd = [compiler, src, opt, "-c", "-o", f"{output_prefix}.o", "-v"]
        backend_result = subprocess.run(backend_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                        encoding='utf-8')
        backend_info = [line for line in backend_result.stdout.splitlines() if
                        "as" in line or "target" in line or "LLVM IR" in line]
        logs[
            "backend"] = f"后端编译命令：{' '.join(backend_cmd)}\n输出目标文件：{output_prefix}.o\n后端关键步骤：\n" + "\n".join(
            backend_info) + "\n"
    except Exception as e:
        logs["backend"] = f"后端编译失败：{str(e)}\n"

    return logs


def compile_deep_analysis(log_file):
    compile_configs = [
        {"name": "O2", "opt": rightOption, "prefix": "o2_deep"},
        {"name": "O3", "opt": wrongOption, "prefix": "o3_deep"}
    ]

    # 日志头部
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"深度编译分析日志 | BugID: {bugid} | 编译器版本：clang 3.3\n")
        f.write(f"测试文件: {fail}\n")
        f.write(f"编译架构: x86-64 | 调试选项: -g | 适配说明：移除clang 3.3不支持的参数\n")
        f.write("=" * 100 + "\n\n")

    for cfg in compile_configs:
        exe_path = f"output_{cfg['name'].lower()}"
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
            stage_logs = get_detailed_compile_logs(compiler, fail, cfg["opt"], cfg["prefix"])
            run_result = run_executable(exe_abs_path) if (compile_success and exe_exists) else {"success": False,
                                                                                                "exit_code": -3,
                                                                                                "output": "",
                                                                                                "error": "编译失败或可执行文件不存在"}
            time_report = [line for line in compile_result.stdout.splitlines() if
                           "Time Report" in line or "Pass" in line or "Total" in line]
            time_report_str = "\n".join(time_report) if time_report else "无详细优化耗时日志"

            # 写入当前配置日志
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"【{cfg['name']} 深度分析】\n")
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

                f.write(f"2. 预处理阶段\n")
                f.write(f"   {stage_logs['preprocess']}\n")
                f.write(f"3. AST分析阶段\n")
                f.write(f"   {stage_logs['ast']}\n")
                f.write(f"4. IR生成与优化阶段\n")
                f.write(f"   {stage_logs['ir_raw']}\n")
                f.write(f"   {stage_logs['ir_opt']}\n")
                f.write(
                    f"   优化pass详情：\n   {stage_logs['opt_passes'].replace('Running pass', '  - Running pass')}\n\n")  # 缩进优化
                f.write(f"5. 汇编生成阶段\n")
                f.write(f"   {stage_logs['asm']}\n")
                f.write(f"6. 后端编译阶段\n")
                f.write(f"   {stage_logs['backend']}\n")
                f.write(f"7. 优化耗时详情\n")
                f.write(f"   {time_report_str}\n")
                f.write("=" * 100 + "\n\n")

        except Exception as e:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"【{cfg['name']} 分析异常】\n")
                f.write(f"   异常信息：{str(e)}\n")
                f.write("=" * 100 + "\n\n")

    # 跨配置差异对比（增强可读性）
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write("【跨配置核心差异对比】\n")
        f.write("=" * 100 + "\n")

        # 1. 优化pass差异（提取O2/O3独有的pass）
        try:
            o2_pass_lines = []
            o3_pass_lines = []
            o2_ll_path = f"{compile_configs[0]['prefix']}_opt.ll"
            o3_ll_path = f"{compile_configs[1]['prefix']}_opt.ll"

            # 重新获取O2/O3的pass列表（避免依赖之前的stage_logs）
            o2_opt_cmd = [compiler, fail, rightOption, "-S", "-emit-llvm", "-o", o2_ll_path, "-debug-pass=Structure"]
            o2_opt_result = subprocess.run(o2_opt_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           encoding='utf-8')
            o2_pass_lines = [line.strip() for line in o2_opt_result.stdout.splitlines() if "Running pass" in line]

            o3_opt_cmd = [compiler, fail, wrongOption, "-S", "-emit-llvm", "-o", o3_ll_path, "-debug-pass=Structure"]
            o3_opt_result = subprocess.run(o3_opt_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           encoding='utf-8')
            o3_pass_lines = [line.strip() for line in o3_opt_result.stdout.splitlines() if "Running pass" in line]

            # 提取独有pass
            o2_only_passes = [p for p in o2_pass_lines if p not in o3_pass_lines]
            o3_only_passes = [p for p in o3_pass_lines if p not in o2_pass_lines]

            f.write(f"1. 优化pass差异\n")
            f.write(f"   O2总pass数：{len(o2_pass_lines)} | O3总pass数：{len(o3_pass_lines)}\n")
            f.write(
                f"   O2独有pass（{len(o2_only_passes)}个）：\n     {chr(10).join(o2_only_passes) if o2_only_passes else '无'}\n")
            f.write(
                f"   O3独有pass（{len(o3_only_passes)}个）：\n     {chr(10).join(o3_only_passes) if o3_only_passes else '无'}\n")
        except Exception as e:
            f.write(f"1. 优化pass差异对比失败：{str(e)}\n")

        # 2. 汇编指令差异（明确标注不同点）
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
                    diff_inst.append(f"{inst}: O2={o2_cnt} | O3={o3_cnt}")

            f.write(f"\n2. 汇编指令差异\n")
            f.write(f"   O2有效指令数：{len(o2_asm)} | O3有效指令数：{len(o3_asm)}\n")
            f.write(f"   指令计数差异：\n     {chr(10).join(diff_inst) if diff_inst else '无'}\n")
            f.write(f"   O2指令分布：{dict(o2_inst_cnt) if o2_inst_cnt else '无'}\n")
            f.write(f"   O3指令分布：{dict(o3_inst_cnt) if o3_inst_cnt else '无'}\n")
        except Exception as e:
            f.write(f"\n2. 汇编差异对比失败：{str(e)}\n")


# ---------------------- 执行日志生成 ----------------------
compile_deep_analysis("deep_compile_analysis.log")
print("深度编译分析日志已生成：deep_compile_analysis.log")