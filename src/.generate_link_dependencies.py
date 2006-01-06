#!/usr/bin/python
import os;
import sys;
import re;
lines=sys.stdin.readlines()
#lines=file('.depend').readlines()
for l in lines:
	if ':' not in l:
		continue
	(so,deps)=re.split(':',l[:-1])
	deps=re.split(' ', deps)
	new_deps=''
	for d in deps:
		if os.path.isfile(os.path.splitext(d)[0]):
			new_deps+=' '+d;
	if (len(new_deps)):
		print so + ':' + new_deps 

#file('b','w').writelines(x)
