#!/bin/bash

docker exec -t devenv /bin/sh \
	-c "cd /opt/shogun/build; cmake -DCMAKE_INSTALL_PREFIX=$HOME/shogun-build -DENABLE_TESTING=ON $CMAKE_OPTIONS .."