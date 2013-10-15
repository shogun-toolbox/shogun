#!/usr/bin/env python
import sys

example_dir='../examples/documented'
target_dir='pages/'

if len(sys.argv) > 1:
	sys.argv[1]
	target_dir=sys.argv[1]

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

		'python_static': ('Static Python', 'ExamplesStaticPython.mainpage',
		'''\nTo run the examples issue
\\verbatim
python name_of_example.py
\\endverbatim
'''
		),

		'octave_static': ('Static Matlab(tm) and Octave', 'ExamplesStaticOctave.mainpage',
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
before starting matlab.
		'''),

		'r_static' : ('Static R', 'ExamplesStaticR.mainpage',
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

		'libshogun': ('C++ libshogun', 'ExamplesLibshogun.mainpage',
		'''\nTo run the examples you will need to manually compile them via
\\verbatim
g++ name_of_example.cpp -lshogun
\\endverbatim

in case you installed libshogun to a nonstandard directory you will need to specify the appropriate library and include paths, e.g.
\\verbatim
g++ -I/path/to/libshogun/includes name_of_example.cpp -L/path/to/libshogun/sofile -lshogun
\\endverbatim

Then the examples are standard binary executables and can be started via
\\verbatim
./name_of_example
\\endverbatim
respectively if the libraries are in nonstandard locations (such that they cannot be found by the dynamic linker)
\\verbatim
LD_LIBRARY_PATH=path/to/libshogun ./name_of_example
\\endverbatim
		'''),

		'cmdline_static' :('Static Command Line', 'ExamplesStaticCmdline.mainpage',
		'''\nTo run the examples issue
\\verbatim
shogun name_of_example.sg
\\endverbatim
		'''
		),

		'lua_modular' : ('Lua Modular', 'ExamplesModularLua.mainpage',
		'''\nTo run the examples issue
\\verbatim
lua name_of_example.lua
\\endverbatim
'''
		),

		'ruby_modular' : ('Ruby Modular', 'ExamplesModularRuby.mainpage',
		'''\nTo run the examples issue
\\verbatim
ruby name_of_example.rb
\\endverbatim
'''
		),

		'csharp_modular' : ('C# Modular', 'ExamplesModularCSharp.mainpage',
		'''\nTo run the examples issue
\\verbatim
gmcs path/to/shogun/interfaces/csharp_modular/*.cs name_of_example.cs
LD_LIBRARY_PATH=path/to/libshogun:path/to/shogun/interfaces/csharp_modular mono name_of_example.exe
\\endverbatim
'''
		),

		'java_modular' : ('Java Modular', 'ExamplesModularJava.mainpage',
		'''\nTo run the examples issue
\\verbatim
javac -jar path/to/modshogun.jar name_of_example.java
java -jar path/to/modshogun.jar name_of_example
\\endverbatim
'''
		),
		}

valid_endings=['.py', '.m', '.R', '.sg', '.cpp', '.lua', '.rb', '.cs', '.java']

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
		if f.find('_') == -1:
			continue

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
