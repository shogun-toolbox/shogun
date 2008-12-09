dump=`echo ${1} | grep WDSVMOcas`
if [ $? -eq 0 ]; then
	echo "WDSVMOcas totally unsupported"
	exit 1
fi

R --no-save --no-restore --no-readline --slave ${1} < test_one.R
