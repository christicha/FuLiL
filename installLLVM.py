# coding=utf-8
import os, random
import datetime
import os.path
import subprocess as subp
import sys


def exccmd(cmd):
    p = os.popen(cmd, "r")
    rs = []
    line = ""
    while True:
        line = p.readline()
        if not line:
            break
        # print line
        # rs.append(line.strip())
    return rs


revisions = [('r256098', 'd206d6c')]
srcdir = '/home/chris/RecBi-workplace/llvmbugs/compilers'
llvm_dir = '/home/chris/llvm-project'  # git上克隆下来llvm，以后使用复制，比较快

for item in revisions:
    rev = item[0]  # 取旧版本号做文件名
    ver = item[1]  # 取新版本号用于checkout
    revpath = srcdir + '/' + rev

    cmake_cmd = (
            'cmake '
            '-DCMAKE_EXPORT_COMPILER_COMMANDS=ON '
            '-DCMAKE_INSTALL_PREFIX=' + revpath + '/' + rev + '-build '
                                                              '-DCMAKE_BUILD_TYPE=Release '
                                                              '-DCMAKE_C_COMPILER=/usr/bin/gcc-7 '
                                                              '-DCMAKE_CXX_COMPILER=/usr/bin/g++-7 '
                                                              '-DCMAKE_C_FLAGS="-g -O0 -fprofile-arcs -ftest-coverage" '
                                                              '-DCMAKE_CXX_FLAGS="-g -O0 -fprofile-arcs -ftest-coverage" '
                                                              '-DCMAKE_EXE_LINKER_FLAGS="-g -fprofile-arcs -ftest-coverage -lgcov" '
                                                              '-DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python2.7 '
                                                              '-DLLVM_ENABLE_ASSERTIONS=ON '
                                                              '-DLLVM_ENABLE_PROJECTS=clang ' +
            revpath + '/' + rev + '/llvm'
    )

    if os.path.exists(revpath):
        exccmd('rm -rf ' + revpath)
    os.system('mkdir ' + revpath)

    os.system('cp -r ' + llvm_dir + ' ' + revpath + '/' + rev)
    os.chdir(revpath + '/' + rev)
    print(revpath + '/' + rev + '\n')
    os.system('git reset --hard HEAD')
    os.system('git checkout ' + ver)
    # before r298721
    # 将clang放入llvm的tools下面
    source_clang = revpath + '/' + rev + '/clang'
    target_tools = revpath + '/' + rev + '/llvm/tools'
    os.system('mkdir -p ' + target_tools)
    os.system('mv ' + source_clang + ' ' + target_tools + '/')
    # 将clang-tools-extra放入指定文件夹
    source_clang_extra = revpath + '/' + rev + '/clang-tools-extra'
    target_clang_tools = revpath + '/' + rev + '/llvm/tools/clang/tools'
    os.system('mkdir -p ' + target_clang_tools)
    os.system('mv ' + source_clang_extra + ' ' + target_clang_tools + '/extra')

    os.system('mkdir ' + revpath + '/' + rev + '-build')
    os.chdir(revpath + '/' + rev + '-build')
    os.system(cmake_cmd)
    os.system('make -j 6')
    os.system('make install')
    os.chdir(srcdir)
