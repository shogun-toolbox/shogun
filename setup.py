#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
# Copyright (c) 2016, Shogun Toolbox Foundation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Written (W) 2016 Ariane Paola Gomes
#

from setuptools import find_packages, setup as shogun_setup
import distutils.command.build
import setuptools.command.install
import distutils.spawn
import glob
import re
import os
import sys

setup_py_location = os.path.abspath(os.path.dirname(__file__))

shogun_build_directory = os.path.join(setup_py_location, 'build')
shogun_generated_install = os.path.join(shogun_build_directory, 'install')
shogun_versionstring_h = os.path.abspath('src/shogun/lib/versionstring.h')
shogun_python_packages_location = None

shogun_completed_cmake = False
shogun_completed_make = False
shogun_completed_make_install = False
show_debug_information = True

with open(os.path.join(setup_py_location, 'README.md')) as f:
  readme = f.read()

try:
    import pypandoc
    readme = pypandoc.convert(readme, to='rst', format='md')
except:
    print("Conversion of long_description from markdown to reStructuredText failed, skipping...")


def shogun_bootstrap():
    global shogun_completed_cmake
    global shogun_completed_make
    global shogun_completed_make_install
    global shogun_python_packages_location

    print("Bootstrapping Shogun")
    
    if shogun_cmake():
        shogun_completed_cmake = True
    else:
        print('Error running CMake')

    if shogun_make():
        shogun_completed_make = True
    else:
        print('Error running make')

    if shogun_make_install():
        shogun_completed_make_install = True
    else:
        print('Error running make install')
    
    if shogun_preconditions_met():
        print("Setting Shogun Python packages location")
        shogun_python_packages_location = glob.glob(os.path.join(shogun_generated_install, 'lib/*/dist-packages'))[0]

        print("Bootstrapping Shogun successfully completed!")
    
        shogun_debug_information()
    else:
        print("Shogun bootstrapping failed!")
        print("Please make sure that you have cmake and make installed.")
        
        sys.exit()


def shogun_preconditions_met():
    print("Verifying Shogun preconditions")
    
    if show_debug_information:
        print("Shogun build environment completed tasks: cmake: [%s] -  make: [%s] - make install: [%s]" % (shogun_completed_cmake, shogun_completed_make, shogun_completed_make_install))
    
    return shogun_completed_cmake and shogun_completed_make and shogun_completed_make_install


def shogun_preconditions():
    if not shogun_preconditions_met():
        shogun_bootstrap()
    
    return shogun_preconditions_met()


def shogun_debug_information():
    if show_debug_information:
        print("Shogun generated installation location %s" % shogun_generated_install)
        print("Shogun Python package location: %s" % shogun_python_packages_location)
        print("Shogun version string location: %s" % shogun_versionstring_h)


def parse_shogun_version(version_header):
    shogun_version_pattern = re.compile(ur'#define MAINVERSION \"([0-9]\.[0-9]\.[0-9])\"')

    with open(version_header, 'r') as f:
        content = f.read()

    matches = re.findall(shogun_version_pattern, content)

    if len(matches):
        return matches[0]
    else:
        return 'undefined'


def shogun_cmake(arguments=None):
    print("Running CMake")

    if arguments is None:
        arguments='-DPythonModular=ON -DENABLE_TESTING=OFF -DCMAKE_INSTALL_PREFIX=install'

    if distutils.spawn.find_executable('cmake') is not None:
        print('CMake arguments: %s ' % arguments)
        print('Creating build directory: %s' % shogun_build_directory)
        distutils.dir_util.mkpath(shogun_build_directory)
        
        os.chdir(shogun_build_directory)
        
        try:
            distutils.spawn.spawn(['cmake'] + arguments.split() + ['..'])
            
        except distutils.spawn.DistutilsExecError:
            print('CMake error.')
            return False
        
        finally:
            os.chdir(os.path.abspath(setup_py_location))
        
        return True
    
    else:
        print('CMake is required to build shogun!')
        return False


def shogun_make(arguments=None):
    print("Running make")

    if arguments is None:
        arguments='all'

    if distutils.spawn.find_executable('make') is not None:
        print('make arguments: %s ' % arguments)

        os.chdir(shogun_build_directory)

        try:
            distutils.spawn.spawn(['make'] + arguments.split())

        except distutils.spawn.DistutilsExecError:
            print('make error.')
            return False

        finally:
            os.chdir(os.path.abspath(setup_py_location))

        return True

    else:
        print('make is required to build shogun!')
        return False


def shogun_make_install():
    return shogun_make(arguments='install')


def get_shogun_version():
    print("Retrieving Shogun version")
    
    if shogun_preconditions():
        shogun_version = parse_shogun_version(shogun_versionstring_h)
        
        if show_debug_information:
            print('The Shogun version is %s' % shogun_version)

        return shogun_version


def python_package_path(package_path):
    print("Generating destination Python package path")
    directories = package_path.split(os.sep)
    destination_path = os.path.join(directories[-2], directories[-1])

    if show_debug_information:
        print("Shogun destination Python package path: %s" % destination_path)
    
    return destination_path


def shogun_packages():
    if not shogun_preconditions_met():
        shogun_bootstrap()

    return find_packages(where=shogun_python_packages_location)


def shogun_package_directories():
    package_directories = dict()
    
    if not shogun_preconditions_met():
        shogun_bootstrap()
    
    package_directories[''] = shogun_python_packages_location
    
    return package_directories


def shogun_data_files():
    data_files = list()
    libshogun_files = glob.glob(os.path.join(shogun_generated_install, 'lib/libshogun*'))
    modshogun_so_destination = os.path.join('lib', python_package_path(shogun_python_packages_location))
    modshogun_so_file = glob.glob(os.path.join(shogun_python_packages_location, '_modshogun.so'))[0]
    
    # appending data files
    data_files.append(('lib', libshogun_files))
    data_files.append((modshogun_so_destination, [modshogun_so_file]))
    
    if show_debug_information: 
        print('Shogun Python package data files:')
        for data_file_content in data_files:
            print('|->[%s]' % data_file_content[0])
            
            for data_file in data_file_content[1]:
                print('    |--> %s' % os.path.basename(data_file))
         
        return data_files


# https://docs.python.org/2/distutils/apiref.html#creating-a-new-distutils-command
class ShogunBuild(distutils.command.build.build):
    user_options = distutils.command.build.build.user_options + [('cmake=', None, 'Specify CMake arguments.')]
    build_base = ''
    build_lib = ''
    build_scripts = ''
    plat_name = ''

    def initialize_options(self):
        self.cmake = None
    
    def finalize_options(self):
        pass
    
    def run(self):
        print('Running Package build')
        
        if not shogun_preconditions_met:
            shogun_cmake(self.cmake)
            shogun_make()
            shogun_make_install()
    
    # Command.sub_commands


class ShogunInstall(setuptools.command.install.install):
    def run(self):
        print('Running Package install')
        
        if shogun_preconditions():
            self.do_egg_install()


shogun_setup(
    name = "shogun-ml",
    version = get_shogun_version(),
    
    description = 'The Shogun Machine Learning Toolbox',
    long_description=readme,
    url = 'http://www.shogun-toolbox.org/',
    
    author = 'Shogun Team',
    author_email = 'shogun-list@shogun-toolbox.org',

    license = 'The GNU General Public License v3.0',
    
    # https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers = [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development :: Libraries',
        # Python 2 and Python 3 support
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 3'
    ],
    
    keywords = [
        'Machine Learning',
        'Gaussian Processes',
        'Neural Networks',
        'Deep Learning'
    ],
    
    zip_safe = False,
    
    # Shogun bootstrapping build and install
    cmdclass = {'build': ShogunBuild, 'install': ShogunInstall},
    
    # Shogun package content
    packages = shogun_packages(),
    package_dir = shogun_package_directories(),
    py_modules =['modshogun'],
    data_files = shogun_data_files(),
    
    # Shogun dependencies
    install_requires = ['numpy']
)

