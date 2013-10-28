import os
import sys

insertions=0
deletions=0
files=0
FROMVER=""
if len(sys.argv)>1:
	FROMVER=sys.argv[1]
TOVER=""
if len(sys.argv)>2:
	TOVER=sys.argv[2]
TMPNAME=os.tmpnam()
VER=""
if len(FROMVER)>0:
	VER=FROMVER+'..'
if len(TOVER)>0:
	if len(VER)==0:
		VER='..'
	VER=VER+TOVER

os.system('git log --oneline --shortstat %s >%s' % (VER,TMPNAME))
for line in file(TMPNAME).readlines():
	if line.find('file') == -1:
		continue
	if line.find('changed') == -1:
		continue
	if line.find('insertion') == -1 and line.find('deletion') == -1:
		continue

	entries=line.split(',')
	for e in entries:
		if e.find('file') != -1:
			files+=int(e.strip().split(' ')[0])
		elif e.find('insertion') != -1:
			insertions+=int(e.strip().split(' ')[0])
		elif e.find('deletion') != -1:
			deletions+=int(e.strip().split(' ')[0])

print "Files changed: %d" % files
print "Insertions: %d" % insertions
print "Deletions: %d" % deletions

os.unlink(TMPNAME)
