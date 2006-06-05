#!/usr/bin/env python

import popen2
import pexpect
import os

os.system('killall R')
p = pexpect.spawn ('R --no-save --no-init-file -d gdb')
p.expect ('(gdb) ')
p.sendline ('set breakpoint pending on\n')
p.expect ('(gdb)')
p.sendline ('run\n')
#p.sendline ('\n')
p.expect ('>')
#libname = 'shogun-Linux-x86_64.debug'
#libname = 'sg.so'
libname = 'shogun-Linux-i686.debug'
loadCmd = 'dyn.load(\"%s\")\n' % (libname)
p.sendline ( loadCmd ) 
p.expect ('>')

#
stdout, stdin = popen2.popen2("ps -A | grep R")
line = stdout.readline()
pid = (line.split())[0]
#
os.system('kill -s PWR ' + str(pid))

#p.sendline ('')
p.expect ('(gdb)')
p.sendline ('b GUIR.cpp:945\n')
p.expect ('(gdb)')
p.sendline ('signal 0\n')
p.expect ('>')
p.sendline ('\n')
p.expect ('>')
#p.interact()
#""".External("sg","send_command","clean_features TRAIN")""", 
#""".External("sg","send_command","clean_features TEST")""",
#""".External("sg","set_features", "TRAIN", traindat)""",
#""".External("sg","set_labels", "TRAIN", trainlab)""", 

commands = [ 
"""traindat <- array(c(1,2,2,1,4,8,5,7),dim=c(4,2))""",
"""trainlab <- array(c(1,1,-1,-1),dim=c(1,4))""",
""".External("sg","set_features", "TRAIN", traindat)""",
"""feat <- .External("sg","get_features", "TRAIN")""", 
""".External("sg","set_labels", "TRAIN", trainlab)""", 
"""lab <- .External("sg","get_labels", "TRAIN")""", 
""".External("sg","send_command", "loglevel ALL")""",
""".External("sg","send_command","clean_kernels")""",
""".External("sg","send_command", "use_linadd 1" )""",
""".External("sg","send_command", "set_kernel GAUSSIAN REAL 50 10")""",
""".External("sg","send_command", "init_kernel TRAIN")""",
"""kt <- .External("sg", "get_kernel_matrix")""",
""".External("sg","send_command", "new_svm LIGHT")""",
""".External("sg","send_command", "c 10.0")""",
""".External("sg","send_command", "svm_epsilon 0.1")""",
""".External("sg","send_command", "svm_train")""",
"""svmAsList <- .External("sg","get_svm")""",
""".External("sg","set_features", "TEST", testdat)""",
""".External("sg","set_labels", "TEST", testlab)""",
""".External("sg","send_command", "init_kernel TEST")""",
"""kte <- .External("sg","get_kernel_matrix")""",
"""out <- .External("sg","svm_classify")""" ]
#"""valerr=mean(testlab~=sign(out))""" ]

counter = 1

for cmd in commands:
   if counter == 16:
      p.interact()
   print "Current command: " + cmd + "\n"
   p.sendline(cmd)
   print p.before
   print p.after
   p.expect ('>')
   counter += 1

p.interact()
