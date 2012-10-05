#!/usr/bin/python

import os

opts=''
#opts=' --prefix=/local/cong --libs=/local/cong/lib --includes=/local/cong/include'

interfaces = ['elwms','cmdline', 'python_modular', 'python', 'r', 'r_modular',
              'octave', 'octave_modular', 'matlab','libshogun','libshogunui', 'perl', 'perl_modular'];

def powerset(s):
    """Return the powerset of a list"""
    d = dict(zip(
        (1<<i for i in range(len(s))),
        (set([e]) for e in s)))

    subset = set()
    yield subset
    for i in range(1, 1<<len(s)):
        subset = subset ^ d[i & -i]
        yield subset


def test_configure():
    """Main"""
    exit_status = []
    for conf in powerset(interfaces):
        conf=tuple(conf)
        interface_string = ','.join(conf)
        p = os.system('./configure --interfaces=' + interface_string + opts + ' >/dev/null 2>&1')
        exit_status.append((conf,p))
        print conf,p
    print exit_status

if __name__ == '__main__':
    test_configure()
