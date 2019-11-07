#!/usr/bin/env python

# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Michele Mazzoni, Sergey Lisitsyn

import sys
if sys.version_info < (3, 0):
    import codecs
    open = codecs.open

# Classes to ignore: mostly because default initialization isn't enough
# to setup the machine for training (i.e. Multitask and DomainAdaptation),
# different reasons are given below.
IGNORE = set([
    # LinearMachines
    'FeatureBlockLogisticRegression', 'LibLinearMTL',
    'MultitaskLinearMachine', 'MultitaskLogisticRegression',
    'MultitaskL12LogisticRegression', 'MultitaskLeastSquaresRegression',
    'MultitaskTraceLogisticRegression', 'MultitaskClusteredLogisticRegression',
    'LatentSVM', 'LatentSOSVM', 'DomainAdaptationSVMLinear',
    'CLinearLatentMachine',

    # KernelMachines
    'DomainAdaptationSVM', 'MKLRegression',
    'MKLClassification', 'MKLOneClass',
    'SVM', # doesn't implement a solver
    'MKL',

    # LinearMulticlassMachines
    'DomainAdaptationMulticlassLibLinear',
    'MulticlassTreeGuidedLogisticRegression',
    'ShareBoost', # apply() takes features subset

    # KernelMulticlassMachines
    'MulticlassSVM', # doesn't implement a solver
    'MKLMulticlass',
    'ScatterSVM', # error C <= 0
    'MulticlassLibSVM' # error C <= 0
])

# Classes that inherit from their template parameter, e.g.
# template <class T> bar : public T { ... }
# We need to check inheritance differently for those
MIXINS = set(["IterativeMachine"])

def read_defined_guards(config_file):
    with open(config_file) as f:
        config = f.read().lower()
        return re.findall('#define (\w+)', config)


def is_guarded(include, defined_guards):
    with open(include, encoding='utf-8') as header:
        guards = re.findall('#ifdef (\w+)', header.read().lower())
        return any([g not in defined_guards for g in guards])


def ignore_in_class_list(include):
    with open(include, encoding='utf-8') as header:
        return 'IGNORE_IN_CLASSLIST' in header.read()


def is_pure_virtual(name, tags):
    return any([name + '\timplementation:pure virtual' in tag for tag in tags])


def use_gpl(path, defined_guards):
    return 'src/gpl/' not in path or 'use_gpl_shogun' in defined_guards


def is_shogun_class(c):
    return c[0] == 'C' and c[1].isupper() and 'class' in c


def get_shogun_classes(tags):
    classes = {}
    # in ctags format it is TAG\tLOCATION\t..\tinherits:CLASS
    for line in filter(is_shogun_class, tags):
        attrs = line.strip().split('\t')
        inherits_str = 'inherits:'
        symbol, location = attrs[0], attrs[1]
        base = attrs[-1][len(inherits_str):] if attrs[-1].startswith(inherits_str) else None

        # parse mixings from declaration, using declarations of the like of
        # /^class CNewtonSVM : public IterativeMachine<LinearMachine>$/;"
        mixin = None
        if base is not None and base in MIXINS:
            mixin = base
            # extract template parameter of mixin and use as base class
            declaration = attrs[-4]
            base = declaration[(declaration.find("<")+1):declaration.find(">")]

        classes[symbol] = {
            'include': location,
            'base': base,
            'mixin': mixin}
    return classes


def get_ancestors(classes, name):
    b = classes[name]['base']
    return [b] + get_ancestors(classes, b) if b in classes else []


def read_ctags(filename):
    if not os.path.exists(filename):
        raise Exception('Failed to found ctags file at %s' % (filename))
    with open(filename) as file:
        return file.readlines()


def generate_tests(input_file, config_file):
    tags = read_ctags(input_file)
    classes = get_shogun_classes(tags)
    guards = read_defined_guards(config_file)

    bases = [
        'LinearMachine', 'KernelMachine', 'LinearMulticlassMachine',
        'KernelMulticlassMachine', 'CNativeMulticlassMachine'
    ]

    # Gather all the machines that inherit from the classes in bases
    machines = {b: {} for b in bases}

    for name, attrs in classes.items():
        ancestors = get_ancestors(classes, name)
        header = attrs['include']
        for base in bases:
            if (base in ancestors) \
                    and not name in IGNORE \
                    and not is_guarded(header, guards) \
                    and not is_pure_virtual(name, tags) \
                    and not ignore_in_class_list(header) \
                    and use_gpl(header, guards):
                machines[base][name] = attrs

    include_template = '#include "{0}"\n'
    typelist_template = 'typedef ::testing::Types<{0}> {1}Types;\n'

    base_test_map = {
        'LinearMachine': 'Machine',
        'CNativeMulticlassMachine': 'Machine',
        'LinearMulticlassMachine': 'Machine',
        'KernelMachine': 'KernelMachine',
        'KernelMulticlassMachine': 'KernelMachine',
    }

    test_machines_map = {
        'Machine': [],
        'KernelMachine': []
    }

    headers = ''
    for b, m in machines.items():
        test_machines_map[base_test_map[b]] += m.keys()
        headers += ''.join([include_template.format(v['include']) for v in m.values()])

    typelists = ''
    for k, v in test_machines_map.items():
        typelists += typelist_template.format(", ".join(v), k)

    return headers + '\n' + typelists


# execution
# ./trained_model_serialization_test.cc.py
# <input file> <output file> <config file>

import sys, os, re
input_file = sys.argv[1]
output_file = sys.argv[2]
config_file = sys.argv[3]

outputText = generate_tests(input_file, config_file)

with open(output_file, 'w') as f:
    f.writelines(outputText)
