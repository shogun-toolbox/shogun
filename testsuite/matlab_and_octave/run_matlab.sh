# !/bin/bash

for file in ../data/*.m; do
        echo -n $file
        echo -n -e "\t\t"
        echo "test_matlab('$file')" | matlab -nojvm -nodisplay | grep '__OK__' > /dev/null
        ret=$?
        if [ $ret -eq 0 ] ; then
                echo 'OK'
        else
                echo 'ERROR'
        fi
done
