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


revisions = ['r199531']
srcdir = '/home/chris/RecBi-workplace/gccbugs/compilers'
gcc_dir = '/home/chris/RecBi-workplace/gccbugs/compilers/r198967/r198967'  # svn下载太慢了，用这个母本去复制

for rev in revisions:

    revpath = srcdir + '/' + rev

    cmake_cmd = (
        revpath + '/' + rev + '/configure '+
        '--enable-languages=c,c++ ' +
        '--disable-werror '+
        '--disable-multilib ' +
        '--enable-checking=release '+
        '--with-gmp=/home/chris/ku/gmp-4.3.2 ' +
        '--with-mpfr=/home/chris/ku/mpfr-3.1.4 ' +
        '--with-mpc=/home/chris/ku/mpc-1.0.3 ' +
        '--prefix=' + revpath + '/' + rev + '-build ' +
        '--enable-coverage '
        '--without-zlib'
    )

    if os.path.exists(revpath):
        exccmd('rm -rf ' + revpath)
    os.system('mkdir ' + revpath)

    os.system('cp -r ' + gcc_dir + ' ' + revpath + '/' + rev)
    os.chdir(revpath + '/' + rev)
    print(revpath + '/' + rev + '\n')
    os.system('svn up -r ' + rev[1:])
    os.system('chmod +x ' + revpath + '/' + rev + '/move-if-change')
    os.system('chmod -R u+rwx ' + revpath + '/' + rev)
    os.system('sed -i \'s/-V/--version/g\' configure')
    os.system('sed -i \'s/-qversion/--version/g\' configure')
    os.system('sed -i \'s/-V/--version/g\' libgcc/configure')
    os.system('sed -i \'s/-qversion/--version/g\' libgcc/configure')
    os.system('find . -name "configure" -type f | xargs sed -i \'s/-V/--version/g\'')
    os.system('find . -name "configure" -type f | xargs sed -i \'s/-qversion/--version/g\'')


    os.system('mkdir ' + revpath + '/' + rev + '-build')
    os.chdir(revpath + '/' + rev + '-build')
    os.system(cmake_cmd)
    os.system('make -j 8')
    os.system('make install')
    os.chdir(srcdir)
