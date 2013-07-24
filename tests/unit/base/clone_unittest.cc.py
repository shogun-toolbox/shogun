#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# Written (W) 2013 Viktor Gal
# Copyright (C) 2013 Viktor Gal

import jinja2
import sys

def get_class_list_content():
    class_list_file = '../../src/shogun/base/class_list.cpp'
    f = open(class_list_file, 'r')
    content = f.readlines()
    f.close()
    return content

def is_valid(sho_class):
    ignores = ['Set', 'ParseBuffer', 'TreeMachine', 'SNPStringKernel', 'WeightedDegreePositionStringKernel']
    if sho_class.startswith('GUI'):
        return False
    if sho_class in ignores:
        return False
    return True

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
                    if is_valid(sho_class):
                        #print l[1]
                        if l[1] == 'SHOGUN_BASIC_CLASS':
                            classes.append(sho_class)
                        if l[1] == 'SHOGUN_TEMPLATE_CLASS':
                            template_classes.append(sho_class)
    return classes, template_classes

templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)

TEMPLATE_FILE = sys.argv[1]
template = templateEnv.get_template(TEMPLATE_FILE)

# get the content of class_list.cpp
class_list_content = get_class_list_content()

classes, template_classes = get_class_list(class_list_content)

types = ['PT_BOOL', 'PT_CHAR', 'PT_INT8', 'PT_UINT8', 'PT_INT16', 'PT_UINT16', 'PT_INT32',
         'PT_UINT32', 'PT_INT64', 'PT_UINT64', 'PT_FLOAT32', 'PT_FLOAT64', 'PT_FLOATMAX']

templateVars = {"classes" : classes, "template_classes" : template_classes, "types" : types}

outputText = template.render(templateVars)

f = open(TEMPLATE_FILE.replace('.jinja2',''), 'w')
f.writelines(outputText)
f.close()

