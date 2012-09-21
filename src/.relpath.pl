#!/usr/bin/env python
#!/usr/bin/env perl
# Based on relpath.py
# Downloaded from http://code.activestate.com/recipes/208993/
# (Thus MIT Licensed)
# Author: Cimarron Taylor
# Date: July 6, 2003
# Program Description: Print relative path from /a/b/c/d to /a/b/c1/d1

import os,sys

def pathsplit(p, rest=[]):
	(h,t) = os.path.split(p)
	if len(h) < 1: return [t]+rest
	if len(t) < 1: return [h]+rest
	return pathsplit(h,[t]+rest)

def commonpath(l1, l2, common=[]):
	if len(l1) < 1: return (common, l1, l2)
	if len(l2) < 1: return (common, l1, l2)
	if l1[0] != l2[0]: return (common, l1, l2)
	return commonpath(l1[1:], l2[1:], common+[l1[0]])

def relpath(p1, p2):
	p1=os.path.expanduser(p1)
	p2=os.path.expanduser(p2)
	p1=os.path.abspath(p1)
	p2=os.path.abspath(p2)
	(common,l1,l2) = commonpath(pathsplit(p1), pathsplit(p2))
	p = []
	if len(l1) > 0:
		s=os.path.pardir + os.path.sep
		p = [ s * len(l1) ]
	p = p + l2
	return os.path.join( *p )

print(relpath(sys.argv[1], sys.argv[2]))
