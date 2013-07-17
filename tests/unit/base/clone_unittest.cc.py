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


def get_class_list(class_list_content):
    import re
    in_class_list = False
    class_list = []
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
                    class_list.append(result.group('sho_class'))
    return class_list

templateLoader = jinja2.FileSystemLoader(searchpath="./")
templateEnv = jinja2.Environment(loader=templateLoader)

TEMPLATE_FILE = sys.argv[1]
template = templateEnv.get_template(TEMPLATE_FILE)

# get the content of class_list.cpp
class_list_content = get_class_list_content()

CLASS_LIST = get_class_list(class_list_content)

templateVars = { "class_list" : CLASS_LIST}

outputText = template.render(templateVars)

f = open(TEMPLATE_FILE.replace('.jinja2',''), 'w')
f.writelines(outputText)
f.close()

