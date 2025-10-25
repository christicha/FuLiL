from Util import findFunByLine,antiModification

file = '/home/chris/FLL-workplace/llvmbugs/info/15920/failcov/method_info.txt'

fileinput = open(file)
lines = fileinput.readlines()
fileinput.close()

for i in range(len(lines)):
    items = lines[i].split(',')
    print(antiModification(items[1]) + '\n')