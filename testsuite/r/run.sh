for file in ../data/*;do 
	echo -n $file
	echo -n -e "\t\t"
	echo "source('test_R.R');test_R('$file')" | R --no-save |grep -q '__OK__'
	ret=$?
	if [ $ret -eq 0 ] ; then
		echo 'OK'
	else
		echo 'ERROR'
	fi
done
