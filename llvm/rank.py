import math
import os
import sys
from configparser import ConfigParser
import re
from collections import defaultdict

from Util import findFunByLine


def deleteGcdaPath(gcdaDirName):
    return re.sub(r'/CMakeFiles/[^/]+.dir', '', gcdaDirName)

def functionRank(bugids, revisions, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    infodir = cfg.get('llvm-locations', 'infodir')
    passdir = cfg.get('llvm-locations', 'passdir')
    resultFile = cfg.get('llvm-locations', 'resultFile')
    resultFile = resultFile.split('.csv')[0] + '_func' + '.csv'
    compilerBasePath = cfg.get('llvm-locations','compilersdir')
    result = open(resultFile, 'w')
    for i in range(len(revisions)):
        rev = revisions[i]  # 当前版本号
        bugId = bugids[i]  # 当前bug ID
        compilerPath = compilerBasePath + revisions[i] + '/' + revisions[i] + '/llvm'
        result.write(rev + '  bug' + bugId + ':\n')  # 写入标题行
        locationfile = open(infodir + bugId + '/locations')
        locationlines = locationfile.readlines()
        locationfile.close()
        buggyfiles = set()
        for i in range(len(locationlines)):
            line = locationlines[i].strip()
            if 'file' in line and 'method' in line:
                buggy_location = line.replace('file:', '').replace('method:', '')
                if '/lib/' in buggy_location:
                    file_part, method_part = buggy_location.split(';', 1)
                    method_part = method_part.rstrip('()')
                    if '/lib/' in file_part:
                        file_part = 'lib/' + file_part.split('/lib/')[1]
                    buggy_location = f"{file_part};{method_part}"
                buggyfiles.add(buggy_location)
        if os.path.exists(infodir + '/' + bugId + '/failcov/stmt_info.txt'):
            stmtpath = infodir + '/' + bugId + '/failcov/stmt_info.txt'
        elif os.path.exists(infodir + '/' + bugId + '/fail/stmt_info.txt'):
            stmtpath = infodir + '/' + bugId + '/fail/stmt_info.txt'
        else:
            print("Error!!")
            sys.exit(1)
        if os.path.exists(infodir + '/' + bugId + '/failcov/method_info.txt'):
            funcpath = infodir + '/' + bugId + '/failcov/method_info.txt'
        elif os.path.exists(infodir + '/' + bugId + '/fail/method_info.txt'):
            funcpath = infodir + '/' + bugId + '/fail/method_info.txt'
        else:
            print("Error!!")
            sys.exit(1)
        # 读取失败用例的语句覆盖率信息
        failfile = open(stmtpath)
        faillines = failfile.readlines()
        failfile.close()
        # 初始化数据结构
        failstmt = dict()  # 记录每个语句被失败用例覆盖的次数
        passstmt = dict()  # 记录每个语句被通过用例覆盖的次数
        failfuncset = set()  # 失败用例覆盖的func集合
        failfuncmapstmt = defaultdict(set)  # 每个文件对应的语句集合映射
        failfileset = set()  # 失败用例覆盖的文件集合
        for i in range(len(faillines)):
            faillinesplit = faillines[i].strip().split(',')
            filename = faillinesplit[0].strip().split('.gcda')[0].strip()  # 提取文件名
            failfileset.add(deleteGcdaPath(filename))
            filename = compilerPath+'/'+deleteGcdaPath(filename)
            if not filename.endswith('.cpp'):  # 只处理.cpp文件
                continue
            stmtlist = faillines[i].strip().split(':')[1].split(',')
            line_to_func = findFunByLine(filename,stmtlist)
            for line,func in line_to_func.items():
                failfuncset.add(func)
                failfuncmapstmt[func].add(line)
                failstmt[func+','+str(line)] = 1
                passstmt[func+','+str(line)] = 0
        # 处理所有通过测试用例的覆盖率
        for test_dir in os.listdir(passdir + '/' + bugId + '/passcov'):  # 修复：使用不同的变量名
            # 读取单个通过用例的覆盖率文件
            passfile = open(passdir + '/' + bugId + '/passcov/' + test_dir + '/stmt_info.txt')
            passlines = passfile.readlines()
            passfile.close()

            # 处理通过用例的覆盖率数据
            for j in range(len(passlines)):
                passlinesplit = passlines[j].strip().split(',')
                filename = passlinesplit[0].strip().split('.gcda')[0].strip()
                filename = deleteGcdaPath(filename)
                if not filename.endswith('.cpp'):  # 只考虑.cpp文件
                    continue
                if filename not in failfileset:  # 只处理失败用例也覆盖的文件
                    continue
                filename = compilerPath + '/' + filename
                stmtlist = passlines[j].strip().split(':')[1].split(',')
                line_to_func = findFunByLine(filename, stmtlist)
                for line, func in line_to_func.items():
                    if f"{func},{line}" in passstmt:
                        passstmt[func+','+str(line)] += 1

        score = dict()
        funcscore = dict()

        for key in failstmt.keys():
            # 核心评分公式：基于失败和通过覆盖率的比值
            score[key] = float(failstmt[key]) / math.sqrt(float(failstmt[key]) * (failstmt[key] + passstmt[key]))
            keyfile = key.split(',')[0]  # 提取文件名
            # 按文件聚合语句得分
            if keyfile not in funcscore.keys():
                funcscore[keyfile] = []
                funcscore[keyfile].append(score[key])
            else:
                funcscore[keyfile].append(score[key])

        #     # 计算每个文件的平均可疑度得分
        #     funcaggstmtscore = dict()
        #     for key in funcscore.keys():
        #         funcaggstmtscore[key] = float(sum(funcscore[key])) / len(funcscore[key])
        #
        #     # 按得分降序排序文件
        #     scorelist = sorted(funcaggstmtscore.items(), key=lambda d: d[1], reverse=True)
        #
        #     # 统计得分为1.0的文件数量（最可疑的文件）
        #     number_1po0 = 0
        #     for j in range(len(scorelist)):
        #         if scorelist[j][0] == 1.0:
        #             number_1po0 += 1
        #
        #     # 处理已知的buggy文件，在排名中找到对应位置
        #     for bf in buggyfiles:
        #         # 尝试匹配文件名（处理路径格式差异）
        #         for j in range(len(scorelist)):
        #             setbf = set(bf.split('/'))
        #             seti = set(scorelist[j][0].split('/'))
        #             if setbf.issubset(seti):  # 如果buggy文件是排名中文件的子集
        #                 bf = scorelist[j][0]  # 使用排名中的完整路径
        #                 break
        #
        #     # 为每个buggy文件计算排名范围
        #     for bf in buggyfiles:
        #         tmp = []
        #         for j in range(len(scorelist)):
        #             if funcaggstmtscore[bf] == scorelist[j][1]:  # 找到得分相同的所有位置
        #                 tmp.append(j)
        #         # 写入结果：文件名,最小排名,最大排名,得分
        #         result.write(
        #             bf + ',' + str(min(tmp) + 1) + ',' + str(max(tmp) + 1) + ',' + str(funcaggstmtscore[bf]) + '\n')
        #
        #     result.write('\n')  # bug之间空行分隔
        #     result.flush()  # 确保数据写入文件
        #
        # result.close()  # 关闭结果文件

        # 计算每个函数的平均可疑度得分
        funcaggstmtscore = dict()
        for key in funcscore.keys():
            funcaggstmtscore[key] = float(sum(funcscore[key])) / len(funcscore[key])

        # 按得分降序排序所有函数
        scorelist = sorted(funcaggstmtscore.items(), key=lambda d: d[1], reverse=True)

        # 写入完整的可疑度排名
        result.write("Rank,Function,Score\n")  # CSV 表头

        for rank, (func_name, score) in enumerate(scorelist, 1):
            result.write(f"{rank},{func_name},{score:.6f}\n")

        # 可选：添加分隔线并标记已知的 buggy 函数
        result.write("\n=== Known Buggy Functions ===\n")
        for bf in buggyfiles:
            # 在完整排名中查找已知 buggy 函数的位置
            found = False
            for rank, (func_name, score) in enumerate(scorelist, 1):
                # 尝试匹配函数名
                setbf = set(bf.split('/'))
                seti = set(func_name.split('/'))
                if setbf.issubset(seti):
                    result.write(f"Buggy Function: {bf} -> Rank: {rank}, Score: {score:.6f}\n")
                    found = True
                    break

            if not found:
                result.write(f"Buggy Function: {bf} -> Not found in ranking\n")

        result.write('\n')  # bug之间空行分隔
        result.flush()  # 确保数据写入文件

        result.close()  # 关闭结果文件


def fileRank(bugids, revisions, configFile):
    cfg = ConfigParser()
    cfg.read(configFile)
    infodir = cfg.get('llvm-locations', 'infodir')
    passdir = cfg.get('llvm-locations', 'passdir')
    resultFile = cfg.get('llvm-locations', 'resultFile')
    resultFile = resultFile.split('.csv')[0] + '_file' + '.csv'
    result = open(resultFile, 'w')

    # 处理每个bug
    for i in range(len(revisions)):
        rev = revisions[i]  # 当前版本号
        bugId = bugids[i]  # 当前bug ID
        result.write(rev + '  bug' + bugId + ':\n')  # 写入标题行

        # 读取bug位置信息文件
        locationfile = open(infodir + bugId + '/locations')
        locationlines = locationfile.readlines()
        locationfile.close()
        buggyfiles = set()  # 存储包含bug的文件集合

        # 解析位置文件，提取bug所在文件
        for i in range(len(locationlines)):
            if 'file' in locationlines[i].strip() and 'method' in locationlines[i].strip():
                # 提取文件路径，格式化为标准格式
                buggyfile = 'lib/' + \
                            locationlines[i].strip().split(';')[0].strip().split(':')[1].strip().split('/lib/')[1]
                buggyfiles.add(buggyfile)  # 添加到bug文件集合

        # 确定失败用例的覆盖率文件路径
        if os.path.exists(infodir + '/' + bugId + '/failcov/stmt_info.txt'):
            tarpath = infodir + '/' + bugId + '/failcov/stmt_info.txt'
        elif os.path.exists(infodir + '/' + bugId + '/fail/stmt_info.txt'):
            tarpath = infodir + '/' + bugId + '/fail/stmt_info.txt'
        else:
            print("Error!!")
            sys.exit(1)

        # 读取失败用例的语句覆盖率信息
        failfile = open(tarpath)
        faillines = failfile.readlines()
        failfile.close()

        # 初始化数据结构
        failstmt = dict()  # 记录每个语句被失败用例覆盖的次数
        passstmt = dict()  # 记录每个语句被通过用例覆盖的次数
        failfileset = set()  # 失败用例覆盖的文件集合
        failfilemapstmt = dict()  # 每个文件对应的语句集合映射

        # 处理失败用例的覆盖率数据
        for i in range(len(faillines)):
            faillinesplit = faillines[i].strip().split(',')
            filename = faillinesplit[0].strip().split('.gcda')[0].strip()  # 提取文件名
            filename = deleteGcdaPath(filename)
            if not filename.endswith('.cpp'):  # 只处理.cpp文件
                continue
            failfileset.add(filename)  # 添加到失败文件集合
            stmtlist = faillines[i].strip().split(':')[1].split(',')  # 提取语句列表
            failfilemapstmt[filename] = set(stmtlist)  # 建立文件到语句的映射

            # 初始化每个语句的计数
            for stmt in stmtlist:
                failstmt[filename + ',' + stmt] = 1  # 失败用例覆盖1次
                passstmt[filename + ',' + stmt] = 0  # 通过用例覆盖0次（初始）

        # 处理所有通过测试用例的覆盖率
        for i in os.listdir(passdir + '/' + bugId + '/passcov'):
            # 读取单个通过用例的覆盖率文件
            passfile = open(passdir + '/' + bugId + '/passcov/' + i + '/stmt_info.txt')
            passlines = passfile.readlines()
            passfile.close()

            # 处理通过用例的覆盖率数据
            for j in range(len(passlines)):
                passlinesplit = passlines[j].strip().split(',')
                filename = passlinesplit[0].strip().split('.gcda')[0].strip()
                filename = deleteGcdaPath(filename)
                if not filename.endswith('.cpp'):  # 只考虑.cpp文件
                    continue
                if filename not in failfileset:  # 只处理失败用例也覆盖的文件
                    continue
                stmtlist = passlines[j].strip().split(':')[1].split(',')
                # 统计通过用例覆盖的语句（只统计失败用例也覆盖的语句）
                for stmt in set(stmtlist) & failfilemapstmt[filename]:
                    passstmt[filename + ',' + stmt] += 1  # 增加通过用例计数

        # 计算每个语句的可疑度得分
        score = dict()
        filescore = dict()

        for key in failstmt.keys():
            # 核心评分公式：基于失败和通过覆盖率的比值
            score[key] = float(failstmt[key]) / math.sqrt(float(failstmt[key]) * (failstmt[key] + passstmt[key]))
            keyfile = key.split(',')[0]  # 提取文件名
            # 按文件聚合语句得分
            if keyfile not in filescore.keys():
                filescore[keyfile] = []
                filescore[keyfile].append(score[key])
            else:
                filescore[keyfile].append(score[key])

        # 计算每个文件的平均可疑度得分
        fileaggstmtscore = dict()
        for key in filescore.keys():
            fileaggstmtscore[key] = float(sum(filescore[key])) / len(filescore[key])

        # 按得分降序排序文件
        scorelist = sorted(fileaggstmtscore.items(), key=lambda d: d[1], reverse=True)

        # 统计得分为1.0的文件数量（最可疑的文件）
        number_1po0 = 0
        for j in range(len(scorelist)):
            if scorelist[j][0] == 1.0:
                number_1po0 += 1

        # 处理已知的buggy文件，在排名中找到对应位置
        for bf in buggyfiles:
            # 尝试匹配文件名（处理路径格式差异）
            for j in range(len(scorelist)):
                setbf = set(bf.split('/'))
                seti = set(scorelist[j][0].split('/'))
                if setbf.issubset(seti):  # 如果buggy文件是排名中文件的子集
                    bf = scorelist[j][0]  # 使用排名中的完整路径
                    break

        # 为每个buggy文件计算排名范围
        for bf in buggyfiles:
            tmp = []
            for j in range(len(scorelist)):
                if fileaggstmtscore[bf] == scorelist[j][1]:  # 找到得分相同的所有位置
                    tmp.append(j)
            # 写入结果：文件名,最小排名,最大排名,得分
            result.write(
                bf + ',' + str(min(tmp) + 1) + ',' + str(max(tmp) + 1) + ',' + str(fileaggstmtscore[bf]) + '\n')

        result.write('\n')  # bug之间空行分隔
        result.flush()  # 确保数据写入文件

    result.close()  # 关闭结果文件
