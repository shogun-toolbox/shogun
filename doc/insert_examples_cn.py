#!/usr/bin/env python
# vim: set fileencoding=utf-8

import sys

example_dir='../examples/documented'
target_dir='pages_cn/'

if len(sys.argv) > 1:
	sys.argv[1]
	target_dir=sys.argv[1]

directories={
		'python_modular' : ('Python模块化接口', 'ExamplesModularPython.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
python name_of_example.py
\\endverbatim
'''
		),

		'r_modular' : ('R模块化接口', 'ExamplesModularR.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
R -f name_of_example.R
\\endverbatim

或者启动R并输入
\\verbatim
source('name_of_example.R')
\\endverbatim
'''
		),

		'octave_modular': ('Octave模块化接口', 'ExamplesModularOctave.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
octave name_of_example.m
\\endverbatim

或者启动octave并输入
\\verbatim
name_of_example
\\endverbatim
'''
		),

		'python_static': ('Python静态接口', 'ExamplesStaticPython.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
python name_of_example.py
\\endverbatim
'''
		),

		'octave_static': ('Matlab(tm)和Octave静态接口', 'ExamplesStaticOctave.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
octave name_of_example.m
\\endverbatim

或者启动octave或Matlab并输入
\\verbatim
name_of_example
\\endverbatim

注意，你要确保sg.oct或sg.mexglx（系统架构不同名字可能不同）已经在 matlab/octave可访问的路径中。
可通过下面的命令添加到它们的路径中：
\\verbatim
addpath /path/to/octave
\\endverbatim
以及
\\verbatim
addpath /path/to/matlab
\\endverbatim

最后请注意，如果是非root用户安装，你需要确保libshogun和libshogunui可以被动态链接器找到，你可能在启动matlab前需要设置：
\\verbatim
LD_LIBRARY_PATH=path/to/libshogun:path/to/libshogunui
\\endverbatim

		'''),

		'r_static' : ('R静态接口', 'ExamplesStaticR.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
R -f name_of_example.R
\\endverbatim

或者启动R并输入
\\verbatim
source('name_of_example.R')
\\endverbatim
'''
		),

		'libshogun': ('C++ libshogun接口', 'ExamplesLibshogun.mainpage',
		'''\n如果要运行这些例子你需要像下面这样编译它们:
\\verbatim
g++ name_of_example.cpp -lshogun
\\endverbatim

如果你是将libshogun安装到非标准目录，你需要指定库目录和头文件目录，例如
\\verbatim
g++ -I/path/to/libshogun/includes name_of_example.cpp -L/path/to/libshogun/sofile -lshogun
\\endverbatim

如果shogun被安装到标准目录，可以像下面这样运行例子
\\verbatim
./name_of_example
\\endverbatim
相反如果被安装到非标准目录（那样它们不能被动态链接器找到），你需要指定
\\verbatim
LD_LIBRARY_PATH=path/to/libshogun ./name_of_example
\\endverbatim
		'''),

		'cmdline_static' :('命令行静态接口', 'ExamplesStaticCmdline.mainpage',
		'''\n要运行这些例子只需要
\\verbatim
shogun name_of_example.sg
\\endverbatim
		'''
			)}

valid_endings=['.py', '.m', '.R', '.sg', '.cpp']

import os
import os.path


for d in directories.keys():
	files=os.listdir(os.path.join(example_dir, d))
	files.sort()

	header=''
	body=''

	old_prefix=None

	header='/*! \page ' + d + '_examples ' + directories[d][0] + '例子\n\n'
	header+='本页面包含了所有' + directories[d][0] + '的例子。\n\n'
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
