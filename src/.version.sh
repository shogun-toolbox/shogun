#!/bin/sh

revision=$(echo '$Id$' | awk -F'M? ' '{ print $3 }')
date=$(echo '$Id$' | awk '{ print $4 }')
time=$(echo '$Id$' | awk -F'Z? ' '{ print $5 }')

year=$(echo $date | awk -F- '{ printf "%d", $1 }')
month=$(echo $date | awk -F- '{ printf "%d", $2 }')
day=$(echo $date | awk -F- '{ printf "%d", $3 }')
hour=$(echo $time | awk -F: '{ printf "%d", $1 }')
minute=$(echo $time | awk -F: '{ printf "%d", $2 }')

extra=""
if test "$1" ; then
	extra="_$1"
fi

echo "#define VERSION_EXTRA \"${extra}\""
echo "#define VERSION_REVISION ${revision}"
echo "#define VERSION_RELEASE \"svn_r${revision}_${date}_${time}${extra}\""
echo "#define VERSION_YEAR ${year}"
echo "#define VERSION_MONTH ${month}"
echo "#define VERSION_DAY ${day}"
echo "#define VERSION_HOUR ${hour}"
echo "#define VERSION_MINUTE ${minute}"
