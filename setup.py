import csv
import os.path
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor
from configparser import ConfigParser
import time

config = ConfigParser()
config.read('./config/config.ini', encoding='utf-8')
compilerBasePath = config.get('llvm-locations', 'compilersdir')
baseResultPath = config.get('llvm-locations', 'resultFile')
buglist = config.get('llvm-locations', 'bugList')
starttime = '2003-12-18'
infoBasePath = config.get('llvm-locations', 'infodir')

bugIds = []
revisions = []
compileOptionRight = []
compileOptionWrong = []

with open(buglist, 'r') as f:
    for line in f:
        line = line.strip()
        items = line.split(',')
        bugIds.append(items[0])
        revisions.append(items[1])
        compileOptionRight.append(items[2])
        compileOptionWrong.append(items[3])


def collectcov(bugid, revision, resultPath):
    covdir = compilerBasePath + revision + '/' + revision + '-build'
    if os.path.exists('gcdalist'):
        os.system('rm gcdalist')
    os.system('find ' + covdir + ' -name \"*.gcda\" > gcdalist')
    f = open('gcdalist')
    lines = f.readlines()
    f.close()
    if not os.path.exists(resultPath):
        os.system('mkdir -p ' + resultPath)
    methodfile = open(resultPath + '/method_info.txt', 'w')
    stmtfile = open(resultPath + '/stmt_info.txt', 'w')

    for i in range(len(lines)):
        os.system('rm *.gcov')
        gcdafile = lines[i].strip()
        if '/clang/test/' in gcdafile:
            continue
        os.system('rm *.gcov')
        if os.path.exists('gcovfile'):
            os.system('rm gcovfile')
        os.system('gcov -f ' + gcdafile + ' > gcovfile')
        if not os.path.exists('./' + gcdafile.strip().split('/')[-1].split('.gcda')[0] + '.gcov'):
            continue

        f = open('gcovfile')
        gcovlines = f.readlines()
        f.close()

        for j in range(len(gcovlines)):
            if 'Function \'' in gcovlines[j].strip():
                if 'Lines executed:' in gcovlines[j + 1].strip() and float(
                        gcovlines[j + 1].strip().split('Lines executed:')[1].split('%')[0].strip()) != 0.0:
                    methodfile.write(
                        gcdafile.split(covdir + '/')[-1] + ',' + gcovlines[j].strip().split('\'')[1] + ',' +
                        gcovlines[j + 1].strip().split('Lines executed:')[1].split('%')[0].strip() + ',' +
                        gcovlines[j + 1].strip().split('of')[-1].strip() + '\n')
        f = open(gcdafile.strip().split('/')[-1].split('.gcda')[0] + '.gcov')
        stmtlines = f.readlines()
        f.close()
        tmp = []
        for j in range(len(stmtlines)):
            if stmtlines[j] == '------------------\n':
                continue
            covcnt = stmtlines[j].strip().split(':')[0].strip()
            linenum = stmtlines[j].strip().split(':')[1].strip()
            if covcnt != '-' and covcnt != '#####':
                tmp.append(linenum)
        if len(tmp) == 0:
            continue
        stmtfile.write(gcdafile.split(covdir + '/')[-1] + ':' + ','.join(tmp) + '\n')


for i in range(len(bugIds)):
    bugid = bugIds[i]
    revision = revisions[i]
    failcovPath = infoBasePath + bugid + '/'
    if not os.path.exists(failcovPath):
        os.makedirs(failcovPath)
    sourcePath = './benchmark/llvmbugs/' + bugid
    for item in os.listdir(sourcePath):
        filePath = os.path.join(sourcePath, item)
        if os.path.isfile(filePath):
            targetPath = os.path.join(failcovPath, item)
            shutil.copy2(filePath, targetPath)
    os.chdir(failcovPath)
    compilerPath = compilerBasePath + revision + '/' + revision + '-build'
    os.system('find ' + compilerPath + ' -name \"*.gcda\" | xargs rm -f')
    os.system(compilerPath + '/bin/clang' + ' ' + 'fail.c')
    resultPath = failcovPath + 'failcov'
    if not os.path.exists(resultPath):
        os.mkdir(resultPath)
    collectcov(bugid, revision, resultPath)