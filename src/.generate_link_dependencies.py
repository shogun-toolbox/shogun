import os,re,sys

incexpr=re.compile('^\s*[%#]include "(\S+)"',re.MULTILINE)
deps=dict();
deps['carrays.i']=[]
deps['cpointer.i']=[]

initial_deps=deps.copy()


def get_deps(f):
	global deps
	global fdep
	for d in deps[f]:
		if d not in fdep:
			fdep+=[d]
			get_deps(d)
			cppfile=d[:-1]+'cpp'

			if cppfile not in fdep and deps.has_key(cppfile):
				fdep+=[cppfile]
				get_deps(cppfile)

#scan each file for potential includes
files=sys.stdin.readlines()
for f in files:
	f=f[:-1]
	deps[f]=re.findall(incexpr, file(f).read())

#generate linker dependencies
for f in deps.iterkeys():
	if f[-1] == 'i' and not initial_deps.has_key(f) and file(f).read().find('%module')>0:
		str=os.path.join(os.path.dirname(f), '_' + os.path.basename(f)[:-2]) + '.so: ' + f[:-2]+'_wrap.cxx.o'

		fdep=list();
		get_deps(f)
		for d in fdep:
			if d[-4:]=='.cpp' or d[-2:]=='.c':
				str+=' ' + d + '.o'

		print str
