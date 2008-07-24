#!/bin/sh

path=${1}

module=`echo ${1}  | awk -F/ '{print $3}'`
filename=`echo ${1}  | awk -F/ '{print $4}' | cut -d '.' -f 1`

echo "test_${module}('${filename}') " | octave
