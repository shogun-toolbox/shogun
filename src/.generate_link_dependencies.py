import os,re,sys

f=None
BASEDIR=None
try:
	f=open('../.config')
	BASEDIR='../'
except:
	f=open('../../.config')
	BASEDIR='../../'

have_doxygen=f.read().find('-DHAVE_DOXYGEN') is not -1

try:
	prefix=sys.argv[1]
	suffix=sys.argv[2]
except IndexError:
	prefix='_'
	suffix='so'

if len(sys.argv)>4 and sys.argv[4]=='external':
	incexpr=re.compile('^\s*[%#]include ("(\S+)")',re.MULTILINE)
else:
	incexpr=re.compile('^\s*[%#]include ("(\S+)"|<shogun/(\S+)>)',re.MULTILINE)

deps=dict();
deps['enumtypeunsafe.swg']=[]
deps['exception.i']=[]
deps['stdint.i']=[]
deps['modshogun_doxygen.i']=[]

modular_deps=['modshogun_doxygen.i']

initial_deps=deps.copy()

def get_deps(f):
	global deps
	global fdep
	try:
		for d in deps[f]:
			if d not in fdep:
				fdep+=[d]
				get_deps(d)
				cppfile=d[:-1]+'cpp'

				if cppfile not in fdep and cppfile in deps:
					fdep+=[cppfile]
					get_deps(cppfile)
	except KeyError:
		print >>sys.stderr, "generate_link_dependencies: warning: could not find dependencies for '%s'" % f 

#scan each file for potential includes
files=sys.stdin.readlines()
for f in files:
	f=f[:-1]
	d=re.findall(incexpr, open(f).read())
	dd=[]
	for i in d:
		if i[1]:
			i=i[1]
			if i.endswith('.h'):
				i=BASEDIR + 'shogun/' + i
		else:
			i=BASEDIR + 'shogun/' + i[2]

		dd.append(i)

	deps[f]=dd
	#print f,deps[f]
	#import pdb
	#pdb.set_trace()

#'./../modular'
#generate linker dependencies
for f in deps.keys():
	if f[-1] == 'i' and not f in initial_deps:
		if open(f).read().find('%module')>-1:
			str1=os.path.join(os.path.dirname(f), prefix + os.path.basename(f)[:-2]) + suffix + ': ' + f[:-2]+'_wrap.cxx.o' + ' sg_print_functions.cpp.o'
			str2=os.path.join(os.path.dirname(f), os.path.basename(f)[:-2]) + '_wrap.cxx: ' + f

			if have_doxygen:
				str2+=' ' + ' '.join([os.path.join(os.path.dirname(f), m) for m in modular_deps])

			fdep=list();
			#if not f.startswith('./../modular/'):
			#	get_deps('./../modular/' + f)
			#else:
			get_deps(f)
			for d in fdep:
				if not d in initial_deps:
					if d[-4:]=='.cpp' or d[-2:]=='.c':
						str1+=' ' + d + '.o'
					if d[-2:]=='.h' or d[-2:]=='.i':
						str2+=' ' + d 

			print(str1)
			print(str2)
