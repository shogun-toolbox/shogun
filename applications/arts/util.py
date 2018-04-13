# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Soeren Sonnenburg, Gunnar Raetsch

import sys
from numpy import mat, array, inf, any, reshape, int32

def parse_name(line, name):
    if (line.startswith(name)):
        sys.stdout.write('.'); sys.stdout.flush()
        return line[line.find('=') + 1:-1]
    else:
        return None

def parse_int(line, name):
    if (line.startswith(name)):
        sys.stdout.write('.'); sys.stdout.flush()
        return int(line[line.find('=') + 1:-1])
    else:
        return None

def parse_float(line, name):
    if (line.startswith(name)):
        sys.stdout.write('.'); sys.stdout.flush()
        return float(line[line.find('=') + 1:-1])
    else:
        return None

def parse_vector(line, file, name):
    mat = parse_matrix(line, file, name)
    if mat is None:
     return mat
    else:
     mat = array(mat).flatten()
     return mat

def parse_string(line, file, name):
    if (line.startswith(name)):
        sys.stdout.write('.'); sys.stdout.flush()
        l = ''
        lines = []
        while l is not None and l.find(']') < 0:
            if l:
                lines.append(l[:-1])
            l = file.readline()

        if l.find(']') < 0:
            sys.stderr.write("string ended without ']'\n")
            return None
        else:
            return lines
    else:
        return None

def parse_matrix(line, file, name):
    if (line.startswith(name)):
        sys.stdout.write('.'); sys.stdout.flush()
        if line.find(']') < 0:
            l = ''
            while l is not None and l.find(']') < 0:
                line += l
                l = file.readline()
            if l is not None and l.find(']') >= 0:
                line += l

        if line.find(']') < 0:
            sys.stderr.write("matrix `" + name + "' ended without ']'\n")
            return None
        else:
            mm = mat(line[line.find('['):line.find(']') + 1])
            if len(mm.shape) == 1:
                mm = reshape(mm.shape[0], 1)
            return mm
    else:
        return None
