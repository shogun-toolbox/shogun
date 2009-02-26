#!/usr/bin/python

import os

interfaces = ['cmdline', 'python_modular', 'python', 'r', 'r_modular',
              'octave', 'octave_modular', 'matlab','libshogun','libshogunui'];

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
        interface_string = ','.join(conf)
        p = os.system('./configure --interfaces='+interface_string
                      +' --prefix=/local/cong --libs=/local/cong/lib --includes=/local/cong/include')
        exit_status.append(p)
    print exit_status

if __name__ == '__main__':
    test_configure()
