from configparser import ConfigParser
from llvm.generateMutate import generateMutate
from llvm.rank import fileRank

config = ConfigParser()
config.read('./config/config.ini', encoding='utf-8')
compilerBasePath = config.get('llvm-locations', 'compilersdir')
baseResultPath = config.get('llvm-locations', 'resultFile')
buglist = config.get('llvm-locations', 'bugList')
passBasePath = config.get('llvm-locations', 'passdir')
configPath = config.get('llvm-locations', 'configFile')

bugIds = []
revisions = []
compileOptionRights = []
compileOptionWrongs = []
checks = []

with open(buglist, 'r') as f:
    for line in f:
        line = line.strip()
        items = line.split(',')
        bugIds.append(items[0])
        revisions.append(items[1])
        compileOptionRights.append(items[2])
        compileOptionWrongs.append(items[3])
        checks.append(items[4])

for i in range(len(bugIds)):
    bugid = bugIds[i]
    revision = revisions[i]
    compileOptionRight = compileOptionRights[i]
    compileOptionWrong = compileOptionWrongs[i]
    check = checks[i]
    failcovPath = passBasePath + bugid + '/'
    generateMutate(bugid, revision, check, compileOptionRight, compileOptionWrong, configPath)
    fileRank(bugid, revision, configPath)