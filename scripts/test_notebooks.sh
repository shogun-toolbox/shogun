#!/bin/bash
#
# This software is distributed under BSD 3-clause license (see LICENSE file).
#
# Authors: Gil Hoben
#

# Known issues in notebooks
#  * Scene_classification.ipynb: issue with cv2 SIFT, causes SEGFAULT
#  * structure/: still has to be ported
skip_files="template.ipynb *-checkpoint.ipynb *.nbconvert.ipynb Scene_classification.ipynb Binary_Denoising.ipynb FGM.ipynb multilabel_structured_prediction.ipynb"

files=$(find $1 -type f -name "*.ipynb" $(printf "! -name %s " $skip_files))
original_dir=$(realpath $1)
declare -a failed_notebooks

num_procs=$2
num_jobs="\j"

process_notebook() {
	cd $(dirname $1)
	logfile=logs_$(basename $1).txt
	jupyter nbconvert --ExecutePreprocessor.timeout=1800 --to notebook --execute $(basename $1) &> $logfile
	return_value=$?
	if [ $return_value -ne 0 ]; then
	    echo "Error $return_value when executing $1."
	    failed_notebooks+=($1)
	    cat $logfile
	else
		echo "Successfully executed $1."
	fi
	rm $logfile
	cd $original_dir
}

for file in $files; do
	while (( ${num_jobs@P} >= num_procs )); do
		wait -n
	done
	process_notebook $file &
done

if [ ${#failed_notebooks[@]} -gt 0 ]; then
	for file in "${failed_notebooks[@]}"; do
	   echo "$file failed"
	done
	exit 1
fi