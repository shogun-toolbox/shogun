#!/usr/bin/env python

# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Viktor Gal, Sergey Lisitsyn, Soeren Sonnenburg, Soumyajit De,
#          Björn Esser, Heiko Strathmann,  Michele Mazzoni


def get_class_list_content(class_list_file):
    f = open(class_list_file, 'r')
    content = f.readlines()
    f.close()
    return content

def get_class_list(class_list_content):
    import re
    in_class_list = False
    classes = []
    template_classes = []
    for line in class_list_content:
            if line == '\n':
                continue
            l = [l.strip() for l in line.split()]
            if 'class_list[]' in l:
                in_class_list = True
                continue

            if in_class_list:
                if '};' in l:
                    in_class_list = False
                    continue
                result = re.match(r"{\"(?P<sho_class>\w+)\"", l[0])

                if result:
                    sho_class=result.group('sho_class')
                    #print l[1]
                    if l[1] == 'SHOGUN_BASIC_CLASS':
                        classes.append(sho_class)
                    if l[1] == 'SHOGUN_TEMPLATE_CLASS':
                        template_classes.append(sho_class)
    return classes, template_classes

def entry(templateFile, class_list_file):
    templateLoader = jinja2.FileSystemLoader(searchpath="/")
    templateEnv = jinja2.Environment(loader=templateLoader)

    template = templateEnv.get_template(templateFile)

    # get the content of class_list.cpp
    class_list_content = get_class_list_content(class_list_file)

    classes, template_classes = get_class_list(class_list_content)

    types = ['PT_BOOL', 'PT_CHAR', 'PT_INT8', 'PT_UINT8', 'PT_INT16', 'PT_UINT16', 'PT_INT32',
             'PT_UINT32', 'PT_INT64', 'PT_UINT64', 'PT_FLOAT32', 'PT_FLOAT64', 'PT_FLOATMAX']

    templateVars = {"classes" : classes, "template_classes" : template_classes, "types" : types}

    return template.render(templateVars)

# execution
# ./clone_unittest.cc.py <template file> <output file name> <extra args...>

import sys, jinja2
TEMPLATE_FILE = sys.argv[1]
output_file = sys.argv[2]
class_list_file = sys.argv[3]

import jinja2
outputText = entry(TEMPLATE_FILE, class_list_file)

f = open(output_file, 'w')
f.writelines(outputText)
f.close()
