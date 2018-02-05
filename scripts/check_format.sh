#!/usr/bin/env bash
# This script checks code style by using clang-format
# on a git diff made between a base branch and a test branch,
# which is the one we want to check.
#
# This script was originally inspired by:
# https://github.com/root-project/root/blob/master/.travis.yml

# Unofficial strict bash mode
# See: http://redsymbol.net/articles/unofficial-bash-strict-mode/
set -euo pipefail
IFS=$'\n\t'

function check_shogun_style {

        # Prevent cases in which we are testing branches
        # different from develop
        if [ ! `git branch --list ${2:-}` ]
        then
            echo "Branch ${2:-} does not exists locally. Fetching it."
            git fetch origin "${2:-}:${2:-}"
            
        fi

        BASE_COMMIT=$(git rev-parse ${2:-})


        echo "-----"
        echo "Shogun Style Checker"
        echo "-----"
        echo "Running clang-format-3.8 against branch ${2:-}, with hash $BASE_COMMIT"

        COMMIT_FILES=$(git diff --name-only $BASE_COMMIT)
		
        # Use clang-format only on existent files
        LIST=("")
        for file in $COMMIT_FILES
        do
            if [ -f $file ]; then
        	    LIST=("${LIST[@]}" "$file")
            fi
        done

        RESULT_OUTPUT="$(git clang-format --commit $BASE_COMMIT --diff --binary `which clang-format` $LIST)"

        if [ "$RESULT_OUTPUT" == "no modified files to format" ] \
            || [ "$RESULT_OUTPUT" == "clang-format-3.8 did not modify any files" ] \
            || [ "$RESULT_OUTPUT" == "clang-format did not modify any files" ]; then
              echo "clang-format-3.8 passed. \o/"
              echo "-----"
              exit 0
        else
            echo "-----"
            echo "clang-format failed."
            echo "To reproduce it locally please run: "
            echo -e "\t1) git checkout ${1:-}"
            echo -e "\t2) git clang-format --commit $BASE_COMMIT --diff --binary $(which clang-format)"
            echo "To fix the errors automatically please run: "
            echo -e "\t1) git checkout ${1:-}"
            echo -e "\t2) git clang-format --commit $BASE_COMMIT --binary $(which clang-format)"
            echo "-----"
            echo "Style errors found:"
            echo "$RESULT_OUTPUT"
            exit 1
        fi
}

# Check only if we have enough arguments
if [[ $# -ne 2 ]]; then
    echo "Wrong number of parameters supplied!"
    echo "Usage: ./check_format.sh <test_branch> <base_branch>"
    exit 1
fi

# Run the check
check_shogun_style ${1:-} ${2:-}
