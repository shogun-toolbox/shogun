#!/bin/sh

OS=`uname -s`
case "$OS" in
     CYGWIN*|Linux)
	year=`date -r CVS/Entries +%Y 2>/dev/null`
	month=`date -r CVS/Entries +%m 2>/dev/null`
	day=`date -r CVS/Entries +%d 2>/dev/null`
	hour=`date -r CVS/Entries +%H 2>/dev/null`
	minute=`date -r CVS/Entries +%M 2>/dev/null`
	;;
     BSD/OS)
	LS=`ls -lT CVS/Entries`
	year=`echo $LS | awk -F" " '{print $9}'`
	month=`echo $LS | awk -F" " '{print $6}'`
	day=`echo $LS | awk -F" " '{print $7}'`
	hms=`echo $LS | awk -F" " '{print $8}'`
	hour=`echo $hms | awk -F":" '{print $1}'`
	minute=`echo $hms | awk -F":" '{print $2}'`
	;;
     Darwin|*) 
	year=`date +%Y 2>/dev/null`
	month=`date +%m 2>/dev/null`
	day=`date +%d 2>/dev/null`
	hour=`date +%H 2>/dev/null`
	minute=`date +%M 2>/dev/null`
	;;
esac

extra=""
if test "$1" ; then
 extra="_$1"
fi

echo "#define VERSION_EXTRA \"${extra}\""
echo "#define VERSION_RELEASE \"cvs_${year}-${month}-${day}_${hour}:${minute}${extra}\""
echo "#define VERSION_YEAR `echo ${year} | sed 's/^[0]*//g'`"
echo "#define VERSION_MONTH `echo ${month} | sed 's/^[0]*//g'`"
echo "#define VERSION_DAY `echo ${day} | sed 's/^[0]*//g'`"
echo "#define VERSION_HOUR `echo ${hour} | sed 's/^[0]*//g'`"
echo "#define VERSION_MINUTE `echo ${minute} | sed 's/^[0]*//g'`"
