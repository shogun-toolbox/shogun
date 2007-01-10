#!/bin/sh

LC_ALL=C
export LC_ALL

extra=""
if test -d .svn
then
	revision=`svn info | grep "^Last Changed Rev:" | cut -f 2 -d ':' | tr -d ' '`
	year=`svn info | grep "^Last Changed Date:" | cut -f 4 -d ' ' | cut -f 1 -d '-'`
	month=`svn info | grep "^Last Changed Date:" | cut -f 4 -d ' ' | cut -f 2 -d '-'`
	day=`svn info | grep "^Last Changed Date:" | cut -f 4 -d ' ' | cut -f 3 -d '-'`
	hour=`svn info | grep "^Last Changed Date:" | cut -f 5 -d ' ' | cut -f 1 -d ':'`
	minute=`svn info | grep "^Last Changed Date:" | cut -f 5 -d ' ' | cut -f 2 -d ':'`

	date="$year-$month-$day"
	time="$hour:$minute"
else
	extra="UNKNOWN_VERSION"
	revision=9999
	date="9999-99-99"
	time="99:99"

	year="9999"
	month="99"
	day="99"
	hour="99"
	minute="99"
fi

if test "$1" ; then
	extra="_$1"
fi

echo "#define VERSION_EXTRA \"${extra}\""
echo "#define VERSION_REVISION ${revision}"
echo "#define VERSION_RELEASE \"svn_r${revision}_${date}_${time}_${extra}\""
echo "#define VERSION_YEAR `echo ${year} | sed 's/^[0]//g'`"
echo "#define VERSION_MONTH `echo ${month} | sed 's/^[0]//g'`"
echo "#define VERSION_DAY `echo ${day} | sed 's/^[0]//g'`"
echo "#define VERSION_HOUR `echo ${hour} | sed 's/^[0]//g'`"
echo "#define VERSION_MINUTE `echo ${minute} | sed 's/^[0]//g'`"
