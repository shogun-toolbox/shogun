#!/usr/bin/env bash

path=${1}
interface=${2-octave}

module=`echo ${1}  | awk -F/ '{print $3}'`
filename=`echo ${1}  | awk -F/ '{print $4}' | cut -d '.' -f 1`

if [ "${interface}" == "octave" ]; then
	echo "${module}('${filename}')" | octave
else
	echo "${module}('${filename}')" | matlab -nojvm -nodisplay
fi
