#!/usr/bin/env bash
# This script was originally inspired by:
# https://github.com/root-project/root/blob/master/.travis.yml

if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then

    BASE_COMMIT=$(git rev-parse $TRAVIS_BRANCH)

    echo "-----"
    echo "Shogun Style Checker"
    echo "-----"
    echo "Running clang-format-3.8 against branch $TRAVIS_BRANCH, with hash $BASE_COMMIT"

    COMMIT_FILES=$(git diff --name-only $BASE_COMMIT)
    RESULT_OUTPUT="$(git clang-format-3.8 --commit $BASE_COMMIT --diff --binary `which clang-format-3.8` $COMMIT_FILES)"

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
        echo -e "\t1) git checkout $TRAVIS_PULL_REQUEST_BRANCH"
        echo -e "\t2) git clang-format-3.8 --commit $BASE_COMMIT --diff --binary $(which clang-format-3.8)"
        echo "To fix the errors automatically please run: "
        echo -e "\t1) git checkout $TRAVIS_PULL_REQUEST_BRANCH"
        echo -e "\t2) git clang-format-3.8 --commit $BASE_COMMIT --binary $(which clang-format-3.8)"
        echo "-----"
        echo "Style errors found:"
        echo "$RESULT_OUTPUT"
        exit 1
    fi
fi
