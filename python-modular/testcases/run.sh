# !/bin/bash

for file in mfiles/*; do
	echo -n $file
	echo -n -e "\t\t"

	python test.py $file > /dev/null
	ret=$? 

	if [ $ret -eq 0 ] ; then
		echo 'OK'
	else
		echo 'ERROR'
	fi
done
