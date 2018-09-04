#!/bin/bash

if [ $CC == "gcc" ] && [ -z ${INTERFACE_OCTAVE} ]; then
  docker exec -t devenv /bin/sh -c "cd /opt/shogun/;  if [ "$TRAVIS_PULL_REQUEST" != "false" ]; then ./scripts/check_format.sh "$TRAVIS_PULL_REQUEST_BRANCH" "$TRAVIS_BRANCH"; fi"
fi

if [ -z ${CMAKE_OPTIONS+x} ]; then
	BUILD_CMD="cd /opt/shogun/build; make -j3"
else
	BUILD_CMD="cd /opt/shogun/build; make -j2"
fi

docker exec -t devenv /bin/sh -c $BUILD_CMD
docker exec -t devenv /bin/sh -c "cd /opt/shogun/build; make install"
docker exec -t devenv /bin/sh -c "cd /opt/shogun/build; ctest --output-on-failure -j 2"
