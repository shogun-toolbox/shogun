#!/bin/sh

path=${1}

module=`echo ${1}  | awk -F/ '{print $3}'`
filename=`echo ${1}  | awk -F/ '{print $4}' | cut -d '.' -f 1`

dump=`echo ${filename} | grep WDSVMOcas`
if [ $? -eq 0 ]; then
	echo "WDSVMOcas totally unsupported"
	echo "ans = 0" # matlab compat hack
	exit 1
fi

echo "${module}('${filename}') " | octave
