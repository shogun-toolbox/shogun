import os,re,sys

try:
	prefix=sys.argv[1]
	suffix=sys.argv[2]
except IndexError:
	prefix='_'
	suffix='so'

incexpr=re.compile('^\s*[%#]include ("(\S+)"|<shogun/(\S+)>)',re.MULTILINE)
deps=dict();
deps['carrays.i']=[]
deps['cpointer.i']=[]
deps['exception.i']=[]
deps['stdint.i']=[]
deps['lib/Library_doxygen.i']=[]
deps['features/Features_doxygen.i']=[]
deps['classifier/Classifier_doxygen.i']=[]
deps['structure/Structure_doxygen.i']=[]
deps['regression/Regression_doxygen.i']=[]
deps['kernel/Kernel_doxygen.i']=[]
deps['preproc/PreProc_doxygen.i']=[]
deps['distribution/Distribution_doxygen.i']=[]
deps['classifier/Classifier_doxygen.i']=[]
deps['clustering/Clustering_doxygen.i']=[]
deps['distrance/Distance_doxygen.i']=[]
deps['evaluation/Evaluation_doxygen.i']=[]
deps['Library_doxygen.i']=[]
deps['Features_doxygen.i']=[]
deps['Classifier_doxygen.i']=[]
deps['Structure_doxygen.i']=[]
deps['Regression_doxygen.i']=[]
deps['Kernel_doxygen.i']=[]
deps['PreProc_doxygen.i']=[]
deps['Distribution_doxygen.i']=[]
deps['Classifier_doxygen.i']=[]
deps['Clustering_doxygen.i']=[]
deps['Distance_doxygen.i']=[]
deps['Evaluation_doxygen.i']=[]

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

				if cppfile not in fdep and deps.has_key(cppfile):
					fdep+=[cppfile]
					get_deps(cppfile)
	except KeyError:
		print >>sys.stderr, "generate_link_dependencies: warning: could not find dependencies for '%s'" % f 

#scan each file for potential includes
files=sys.stdin.readlines()
for f in files:
	f=f[:-1]
	d=re.findall(incexpr, file(f).read())
	dd=[]
	for i in d:
		if i[1]:
			i=i[1]
			if i.endswith('.h'):
				i='../shogun/' + i
		else:
			i='../shogun/' + i[2]

		dd.append(i)

	deps[f]=dd
	#print f,deps[f]
	#import pdb
	#pdb.set_trace()
#'./../modular'
#generate linker dependencies
for f in deps.iterkeys():
	if f[-1] == 'i' and not initial_deps.has_key(f):
		if file(f).read().find('%module')>-1:
			str1=os.path.join(os.path.dirname(f), prefix + os.path.basename(f)[:-2]) + '.' + suffix + ': ' + f[:-2]+'_wrap.cxx.o'
			str2=os.path.join(os.path.dirname(f), os.path.basename(f)[:-2]) + '_wrap.cxx: ' + f

			fdep=list();
			#if not f.startswith('./../modular/'):
			#	get_deps('./../modular/' + f)
			#else:
			get_deps(f)
			for d in fdep:
				if not initial_deps.has_key(d):
					if d[-4:]=='.cpp' or d[-2:]=='.c':
						str1+=' ' + d + '.o'
					if d[-2:]=='.h' or d[-2:]=='.i':
						str2+=' ' + d 

			print str1
			print str2
