#!/bin/sh

LC_ALL=C
export LC_ALL

prefix=""
mainversion=`awk '/Release/{print $5;exit}' ../NEWS`
parameter=`awk '/Release/{print $11;exit}' ../NEWS`
extra=""
if test -d .svn
then
	revision=`svn info | grep "^Last Changed Rev:" | cut -f 2 -d ':' | tr -d ' '`
	year=`svn info | grep "^Last Changed Date:" | cut -f 4 -d ' ' | cut -f 1 -d '-'`
	month=`svn info | grep "^Last Changed Date:" | cut -f 4 -d ' ' | cut -f 2 -d '-'`
	day=`svn info | grep "^Last Changed Date:" | cut -f 4 -d ' ' | cut -f 3 -d '-'`
	hour=`svn info | grep "^Last Changed Date:" | cut -f 5 -d ' ' | cut -f 1 -d ':'`
	minute=`svn info | grep "^Last Changed Date:" | cut -f 5 -d ' ' | cut -f 2 -d ':'`

	src="svn"
	prefix="svn_r"
elif test -d ../../.git
then
	branch_point=$(git merge-base master HEAD)
	dateinfo=$(git show --pretty='format:%ai' $branch_point | head -1)

	year=$(echo $dateinfo | cut -f 1 -d '-')
	month=$(echo $dateinfo | cut -f 2 -d '-')
	day=$(echo $dateinfo | cut -f 3 -d '-' | cut -f 1 -d ' ')
	hour=$(echo $dateinfo | cut -f 2 -d ' ' | cut -f 1 -d ':')
	minute=$(echo $dateinfo | cut -f 2 -d ' ' | cut -f 2 -d ':')

	revision="`git show --pretty='format:%h'|head -1`"
	revision_prefix="0x"
	prefix="git_"
else
	extra="UNKNOWN_VERSION"
	revision=9999

	year="9999"
	month="99"
	day="99"
	hour="99"
	minute="99"
	src="custom"
fi

date="$year-$month-$day"
time="$hour:$minute"

if test "$1" ; then
	extra="_$1"
fi

echo "#define MAINVERSION \"${mainversion}\""

echo "#define VERSION_EXTRA \"${extra}\""
echo "#define VERSION_REVISION ${revision_prefix}${revision}"
echo "#define VERSION_RELEASE \"${prefix}${revision}_${date}_${time}_${extra}\""
echo "#define VERSION_YEAR `echo ${year} | sed 's/^[0]//g'`"
echo "#define VERSION_MONTH `echo ${month} | sed 's/^[0]//g'`"
echo "#define VERSION_DAY `echo ${day} | sed 's/^[0]//g'`"
echo "#define VERSION_HOUR `echo ${hour} | sed 's/^[0]//g'`"
echo "#define VERSION_MINUTE `echo ${minute} | sed 's/^[0]//g'`"
echo "#define VERSION_PARAMETER `echo ${parameter} | sed 's/\(.*\)./\1/'`"
