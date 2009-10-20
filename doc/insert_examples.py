#!/usr/bin/env python

example_dir='../examples'
target_dir='pages/'
directories={
		'python_modular' : ('Python Modular', 'ExamplesModularPython.mainpage',
		'''\nTo run the examples issue
\\verbatim
python name_of_example.py
\\endverbatim
'''
		),

		'r_modular' : ('R Modular', 'ExamplesModularR.mainpage',
		'''\nTo run the examples issue
\\verbatim
R -f name_of_example.R
\\endverbatim

or start R and then type
\\verbatim
source('name_of_example.R')
\\endverbatim
'''
		),

		'octave_modular': ('Octave Modular', 'ExamplesModularOctave.mainpage',
		'''\nTo run the examples issue
\\verbatim
octave name_of_example.m
\\endverbatim

or start up octave and then type
\\verbatim
name_of_example
\\endverbatim
'''
		),

		'python': ('Static Python', 'ExamplesStaticPython.mainpage',
		'''\nTo run the examples issue
\\verbatim
python name_of_example.m
\\endverbatim
'''
		),

		'octave': ('Static Matlab(tm) and Octave', 'ExamplesStaticOctave.mainpage',
		'''\nTo run the examples issue
\\verbatim
octave name_of_example.m
\\endverbatim

or start up matlab or octave and then type
\\verbatim
name_of_example
\\endverbatim

Note that you have to make sure that the sg.oct or sg.mexglx (name varies with architecture)
has to be in the matlab/octave path. This can be achieved using the addpath command:
\\verbatim
addpath /path/to/octave
\\endverbatim
respectively
\\verbatim
addpath /path/to/matlab
\\endverbatim

Finally note that for non-root installations you will have to make sure that libshogun and libshogun ui can be found by the dynamic linker, e.g. you will need to set:

\\verbatim
LD_LIBRARY_PATH=path/to/libshogun:path/to/libshogunui
\\endverbatim
before startign matlab.
		'''),

		'r' : ('Static R', 'ExamplesStaticR.mainpage',
		'''\nTo run the examples issue
\\verbatim
R -f name_of_example.R
\\endverbatim

or start R and then type
\\verbatim
source('name_of_example.R')
\\endverbatim
'''
		),

		'cmdline' :('Static Command Line', 'ExamplesStaticCmdline.mainpage',
		'''\nTo run the examples issue
\\verbatim
shogun name_of_example.sg
\\endverbatim
		'''
			)}

valid_endings=['.py', '.m', '.R', '.sg']

import os
import os.path


for d in directories.keys():
	files=os.listdir(os.path.join(example_dir, d))
	files.sort()

	header=''
	body=''

	old_prefix=None

	header='/*! \page ' + d + '_examples Examples for ' + directories[d][0] + ' Interface\n\n'
	header+='This page lists ready to run shogun examples for the ' + directories[d][0] + ' interface.\n\n'
	for f in files:
		prefix=f[0:f.find('_')]
		suffix=f[f.rfind('.'):]
		#print prefix
		#print suffix
		if suffix not in valid_endings:
			continue
		if prefix != old_prefix:
			old_prefix=prefix
			body+='\n'
			header += '\\li \subpage ' + d + '_' + prefix + '_examples \n'
			body+='\\section ' + d + '_' + prefix + '_examples ' + prefix.title() + '\n\n'

		body += '\n\\li <b>' + os.path.join(example_dir, d, f) + '</b>\n'
		body += '\\verbinclude ' + f + '\n'
	
	header+=directories[d][2]
	text = header + body + '*/'
	file(os.path.join(target_dir, directories[d][1]),'w').write(text)
