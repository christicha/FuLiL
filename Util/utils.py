import os.path
import re
import subprocess

def findCppByGcda(compilerPath, gcdaPath) :
    searchPath = [compilerPath, compilerPath+'/llvm/tools/clang']
    gcdaBaseName = os.path.basename(gcdaPath)
    cppBaseName = re.sub(r'\.gcda$', '', gcdaBaseName)
    gcdaDirName = os.path.dirname(gcdaPath)
    candidatePath = re.sub(r'CMakeFiles/[^/]+.dir', '', gcdaDirName)
    candidatePath = os.path.join(candidatePath, cppBaseName)

    if os.path.exists(candidatePath):
        return os.path.abspath(candidatePath)

    if searchPath:
        for rootDir in searchPath:
            for root, _, files in os.walk(rootDir):
                if cppBaseName in files:
                    return os.path.abspath(os.path.join(root,cppBaseName))


def findFunByLine(cppFilePath, targetLine):
    basePath = re.sub(r'^.*llvm/', '', cppFilePath)
    if os.path.exists('tags'):
        os.system('rm tags')
    cmd = [
        "ctags", "-n", "--c++-kinds=+p", "--fields=+n", "-o", "tags", cppFilePath
    ]
    try:
        subprocess.run(cmd,check=True,capture_output=True,text=True)
    except subprocess.CalledProcessError as e:
        print(f"ctags error: {e.stderr}")
        raise

    func_tags = []
    with open("tags", 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!_TAG_'):
                continue
            parts = line.split('\t')
            if len(parts) < 4:
                continue
            func_name = parts[0]
            line_info = parts[2]
            kind_info = parts[3]

            if kind_info != 'f':
                continue

            start_line = int(re.search(r'(\d+);', line_info).group(1))

            class_name = None
            for part in parts[4:]:
                if part.startswith('class:'):
                    class_name = part.split(':', 1)[1]
                    break

            full_func_name = f"{basePath};{class_name}::{func_name}" if cppFilePath else func_name
            func_tags.append((full_func_name, start_line))

    func_tags.sort(key=lambda x: x[1])
    if not func_tags:
        print("No any function tags")
        return {}

    with open(cppFilePath,'r') as f:
        total_lines = sum(1 for _ in f)

    func_ranges = []

    for i in range(len(func_tags)):
        func_name, start = func_tags[i]
        end = func_tags[i + 1][1] - 1 if i + 1 < len(func_tags) else total_lines
        func_ranges.append((func_name, start, end))

    line_to_fun = {}

    for line in targetLine:
        for func_name,start,end in func_ranges:
            line = int(line)
            if start <= line <= end:
                line_to_fun[line] = func_name

    return line_to_fun

def antiModification(modifyName):
    """调用c++filt工具将修饰的函数名转换为未修饰形式"""
    try:
        result = subprocess.run(
            ['c++filt', modifyName],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return modifyName
    except FileNotFoundError:
        print("c++filt not found")
        return modifyName