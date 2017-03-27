#!/usr/bin/env python

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#

def get_ancestors(c, classes):
    l = []
    b = classes[c]['base']
    while b:
        l.append(b)
        try:
            b = classes[b]['base']
        except:
            break
    return l


def get_descendants(base, classes):
    result = {}
    for name, attrs in classes.items():
        if base in get_ancestors(name, classes):
            result[name] = attrs
    return result


def entry(templateFile, input_file, root_dir, bin_dir):
    templateLoader = jinja2.FileSystemLoader(searchpath="/")
    templateEnv = jinja2.Environment(loader=templateLoader)

    template = templateEnv.get_template(templateFile)

    import json
    with open(input_file) as f:
        classes = json.load(f)
        lms = get_descendants('LinearMachine', classes)

    data_path = os.path.relpath(os.path.join(root_dir, 'data', 'toy'), bin_dir)
    templateVars = {"classes" : lms, "data_path" : data_path}

    return template.render(templateVars)


# execution
# ./trained_model_serialization_unittest.cc.py
# <template file> <input file> <output file> <shogun root dir> <test bin dir>

import sys, os, json
TEMPLATE_FILE = sys.argv[1]
input_file = sys.argv[2]
output_file = sys.argv[3]
root_dir = sys.argv[4]
bin_dir = sys.argv[5]

try:
    import jinja2
    outputText = entry(TEMPLATE_FILE, input_file, root_dir, bin_dir)
except ImportError:
    import os
    basename = os.path.basename(output_file)
    basename = basename.replace('.cc', '')
    print("Please install jinja2 for trained model serialization unit-tests")
    outputText = ['''#include <gtest/gtest.h>
TEST(Dummy, %s_dummy)
{
}''' % (basename)]

f = open(output_file, 'w')
f.writelines(outputText)
f.close()
