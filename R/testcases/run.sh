for file in ../../python-modular/testcases/mfiles/*;do 
	echo -n $file
	echo -n -e "\t\t"
	#echo "source('test_R.R');test_R('$file')" | R --no-save |grep '__OK__'
	echo "source('test_R.R');test_R('$file')" | R --no-save
	exit
	ret=$?
	if [ $ret -eq 0 ] ; then
		echo 'OK'
	else
		echo 'ERROR'
	fi
done
