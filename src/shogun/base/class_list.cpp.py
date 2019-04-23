#!/usr/bin/env python

# Copyright (c) The Shogun Machine Learning Toolbox
# Copyright (c) 2008-2009 Fraunhofer Institute FIRST and Max-Planck-Society
# Written (w) 2008-2009 Soeren Sonnenburg
# Written (w) 2016 - 2017 Heiko Strathmann
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# The views and conclusions contained in the software and documentation are those
# of the authors and should not be interpreted as representing official policies,
# either expressed or implied, of the Shogun Development Team.

import os

class_str = 'class'
types = ["BOOL", "CHAR", "INT8", "UINT8", "INT16", "UINT16", "INT32", "UINT32",
         "INT64", "UINT64", "FLOAT32", "FLOAT64", "FLOATMAX", "COMPLEX128"]
config_tests = ["HAVE_HDF5", "HAVE_LAPACK",
                "USE_CPLEX", "USE_SVMLIGHT", "USE_GLPK", "USE_LZO", "USE_GZIP",
                "USE_BZIP2", "USE_LZMA", "USE_MOSEK", "HAVE_COLPACK",
                "HAVE_NLOPT", "HAVE_PROTOBUF", "HAVE_VIENNACL", "USE_GPL_SHOGUN",
                "USE_META_INTEGRATION_TESTS", "HAVE_TFLOGGER"]
# TODO: remove once plugins are working
class_blacklist = ["SGVector", "SGMatrix", "SGSparseVector", "SGSparseMatrix", 
        "SGStringList", "SGMatrixList", "SGCachedVector", "SGNDArray",
        "ObservedValue", "ObservedValueTemplated", "ParameterObserverCV",
        "ParameterObserverHistogram", "ParameterObserverScalar", "ParameterObserverTensorBoard",
        "TBOutputFormat", "Iterator", "Wrapper", "PIterator",
        "BitPackedFlatHashTableError", "TypedAnyPolicy", "NonOwningAnyPolicy",
        "PointerValueAnyPolicy", "InvalidStateException", "NotFittedException",
        "ShogunException", "BitPackedVectorError", "CompositeHashTableError",
        "DataStorageError", "DataTransformationError", "AugmentedHeap",
        "HashTableError", "LSHTableError", "LSHFunctionError",
        "NearestNeighborQueryError", "StaticProbingHashTableError", "DynamicProbingHashTableError",
        "FalconnError", "LSHNearestNeighborTableError", "LSHNNTableWrapper",
        "Tag", "LinalgBackendGPUBase", "LinalgBackendEigen", "LinalgBackendViennaCL",
        "TypeMismatchException", "FlatHashTableError", "SimpleHeap", "LSHNNTableSetupError",
        "KDTREEKNNSolver", "LDACanVarSolver",
        "GammaFeatureNumberInit", "StdVectorPrefetcher", "ShogunNotImplementedException",
        "ToStringVisitor", "BitseryVisitor", "FileSystem", "PosixFileSystem",
        "WindowsFileSystem", "LocalWindowsFileSystem", "LocalPosixFileSystem",
        "NullFileSystem", "FilterVisitor", "RandomMixin", "MaxCrossValidation",
        "StreamingDataFetcher", "MaxMeasure", "MaxTestPower",
        "MedianHeuristic", "WeightedMaxMeasure", "WeightedMaxTestPower",
        "Seedable", "ShogunEnv"]

SHOGUN_TEMPLATE_CLASS = "SHOGUN_TEMPLATE_CLASS"
SHOGUN_BASIC_CLASS = "SHOGUN_BASIC_CLASS"


def check_class(line):
    if not (line.find('public') == -1 and
            line.find('private') == -1 and
            line.find('protected') == -1):
        return True


def check_abstract_class(line):
    line = line.replace(' ', '').replace('\t', '').strip()
    return line.endswith('=0;')


def check_is_in_blacklist(c, lines, line_nr, blacklist):
    ifdef_cnt = 0
    for i in range(line_nr, 0, -1):
        line = lines[i]
        if line.find('#endif') != -1:
            ifdef_cnt -= 1
        if line.find('#ifdef') != -1:
            ifdef_cnt += 1

            for b in blacklist.keys():
                if line.find(b) != -1 and ifdef_cnt > 0:
                    return True
        if line.find('#ifndef') != -1:
            ifdef_cnt += 1

    return False


def extract_class_name(lines, line_nr, line, blacklist):
    try:
        if not line:
            line = lines[line_nr]
        c = line[line.index(class_str)+len(class_str):]
        if ':' not in c:
            return
        if not check_class(line):
            if not check_class(lines[line_nr+1]):
                return
        c = c.split()[0]
    except:
        return

    c = c.strip(':').strip()

    if c.endswith(';'):
        return
    if '>' in c:
        return
    if not (len(c) > 2 and c[0].isupper()):
        return
    if check_is_in_blacklist(c, lines, line_nr, blacklist) or (c in class_blacklist):
        return
    return c


def get_includes(classes, headers_absolute_fnames):
    includes = []
    for c in classes:
        for h in headers_absolute_fnames:
            class_from_header = os.path.splitext(os.path.basename(h))[0]

            # build relative include path from absolute header filename
            if class_from_header in c:
                # find *last* occurence of "shogun" dir in header
                shogun_dir = "shogun"
                assert shogun_dir in h
                tails = []
                head, tail = os.path.split(h)
                while tail != shogun_dir and len(head)>0:
                    tails += [tail]
                    head, tail = os.path.split(head)

                # construct include path from collected tails
                tails.reverse()
                include = os.path.join(*([shogun_dir] + tails))

                # thats your include header
                includes.append("#include <%s>" % include)

    return includes

def get_definitions(classes):
    definitions = []
    definitions.append("#define %s" % SHOGUN_TEMPLATE_CLASS)
    definitions.append("#define %s" % SHOGUN_BASIC_CLASS)
    for c, t in classes:
        d = "static %s SGObject* __new_%s(EPrimitiveType g) { return g == PT_NOT_GENERIC? new %s(): NULL; }" % (SHOGUN_BASIC_CLASS,c,c)
        definitions.append(d)
    return definitions


def get_template_definitions(classes, supports_complex):
    definitions = []
    for c, t in classes:
        d = []
        d.append("static %s SGObject* __new_%s(EPrimitiveType g)\n{\n\tswitch (g)\n\t{\n"
                 % (SHOGUN_TEMPLATE_CLASS, c))
        for t in types:
            if t in ('BOOL', 'CHAR'):
                suffix = ''
            else:
                suffix = '_t'
            if t == 'COMPLEX128' and not supports_complex:
                d.append("\t\tcase PT_COMPLEX128: return NULL;\n")
            else:
                d.append("\t\tcase PT_%s: return new %s<%s%s>();\n"
                         % (t, c, t.lower(), suffix))
        d.append("\t\tcase PT_SGOBJECT:\n")
        d.append("\t\tcase PT_UNDEFINED: return NULL;\n\t}\n\treturn NULL;\n}")
        definitions.append(''.join(d))
    return definitions


def get_struct(classes):
    struct = []
    for c, template in classes:
        prefix = SHOGUN_BASIC_CLASS
        if template:
            prefix = SHOGUN_TEMPLATE_CLASS

        s = '{"%s", %s __new_%s},' % (c, prefix, c)
        struct.append(s)
    return struct


def extract_block(c, lines, start_line, stop_line, start_sym, stop_sym):
    sym_cnt = 0

    block_start = -1
    block_stop = -1

    for line_nr in range(start_line, stop_line):
        line = lines[line_nr]
        if line.find(start_sym) != -1:
            sym_cnt += 1
            if block_start == -1:
                block_start = line_nr
        if line.find(stop_sym) != -1:
            block_stop = line_nr+1
            sym_cnt -= 1
        if sym_cnt == 0 and block_start != -1 and block_stop != -1:
            return block_start, block_stop

    return block_start, block_stop


def check_complex_supported_class(line):
    l = list(filter(lambda y: y if y != '' else None,
             line.strip().replace('\t', ' ').split(' ')))
    supported = len(l) == 3 and l[0] == 'typedef' and l[1] == 'bool' \
        and l[2] == 'supports_complex128_t;'
    return supported


def test_candidate(c, lines, line_nr, supports_complex):
    start, stop = extract_block(c, lines, line_nr, len(lines), '{', '}')
    if stop < line_nr:
        return False, line_nr+1
    complex_supported = False
    for line_nr in range(start, stop):
        line = lines[line_nr]
        if line.find('virtual') != -1:
            if check_abstract_class(line):
                return False, stop
            else:
                vstart, vstop = extract_block(c, lines, line_nr,
                                              stop, '(', ')')
                for line_nr in range(vstart, vstop):
                    line = lines[line_nr]
                    if check_abstract_class(line):
                        return False, stop
        if line.find('supports_complex128_t') != -1:
            if check_complex_supported_class(line):
                complex_supported = True
                if not supports_complex:
                    return False, stop
    if supports_complex and not complex_supported:
        return False, stop

    return True, stop


def extract_classes(HEADERS, template, blacklist, supports_complex):
    """
    Search in headers for non-template/non-abstract class-names

    Does not support local nor multiple classes and
    drops classes with pure virtual functions
    """
    classes = list()
    for fname in HEADERS:
        if fname.find("external") > 0:
            continue
        try:
            lines = open(fname).readlines()
        except: # python3 workaround
            lines = open(fname, encoding='utf-8', errors='ignore').readlines()
        line_nr = 0
        while line_nr < len(lines):
            line = lines[line_nr]

            if line.find('IGNORE_IN_CLASSLIST') != -1:
                line_nr += 1
                continue
            c = None
            if template:
                tp = line.find('template')
                if tp != -1:
                    line = line[tp:]
                    cp = line.find('>')
                    line = line[cp+1:]
                    cp = line.find(class_str)
                    if cp != -1:
                        c = extract_class_name(lines, line_nr, line, blacklist)
            else:
                if line.find(class_str) != -1:
                    c = extract_class_name(lines, line_nr, None, blacklist)
            if c:
                ok, line_nr = test_candidate(c, lines,
                                             line_nr, supports_complex)
                if ok:
                    classes.append((c, template))
                continue

            line_nr += 1
    return classes


def write_templated_file(fname, substitutes):
    template = open(fname).readlines()

    f = open(fname, 'w')
    for line in template:
        l = line.strip()
        if l.startswith('REPLACE') and l.endswith('THIS'):
            l = line.split()[1]
            if sys.version_info >= (3,):
                for s in substitutes.keys():
                    if l == s:
                        f.write('\n'.join(substitutes[s]))
                    continue
            else:
                for s in substitutes.iterkeys():
                    if l == s:
                        f.write('\n'.join(substitutes[s]))
                    continue
        else:
            f.write(line)


def read_config():
    config = dict()
    for line in open('lib/config.h').readlines():
        if line == '\n' or \
           line.lstrip().startswith("/*") or \
           line.lstrip().startswith("//"):
            continue
        l = [l.strip() for l in line.split()]
        if len(l) > 1:
            config[l[1]] = 1

    return config


def get_blacklist():
    config = read_config()
    blacklist = dict()
    for cfg in config_tests:
        if cfg not in config:
            blacklist[cfg] = 1
    return blacklist

if __name__ == '__main__':
    import sys
    TEMPL_FILE = sys.argv[1]
    HEADERS = None
    if (sys.argv[2] == "-in"):
        # read header file list from file
        with open(sys.argv[3]) as f:
            content = f.readlines()
            HEADERS = [x.strip() for x in content]
    else:
        HEADERS = sys.argv[2:]

    blacklist = get_blacklist()

    classes = extract_classes(HEADERS, False, blacklist, False)
    template_classes = extract_classes(HEADERS, True, blacklist, False)
    complex_template_classes = extract_classes(HEADERS, True, blacklist, True)
    includes = get_includes(classes+template_classes+complex_template_classes, HEADERS)
    definitions = get_definitions(classes)
    template_definitions = get_template_definitions(template_classes, False)
    complex_template_definitions = get_template_definitions(complex_template_classes, True)
    struct = get_struct(classes+template_classes+complex_template_classes)
    substitutes = {'includes': includes,
                   'definitions': definitions,
                   'template_definitions': template_definitions,
                   'complex_template_definitions': complex_template_definitions,
                   'struct': struct
                   }

    write_templated_file(TEMPL_FILE, substitutes)
