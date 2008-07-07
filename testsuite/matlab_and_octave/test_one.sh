#!/bin/sh

path=${1}
is_octave=${2-1}

module=`echo ${1}  | awk -F/ '{print $3}'`
filename=`echo ${1}  | awk -F/ '{print $4}' | cut -d '.' -f 1`

if [ ${is_octave} -eq 1 ]; then
	echo "${module}('${filename}')" | octave
else
	echo "${module}('${filename}')" | matlab -nojvm -nodisplay
fi
