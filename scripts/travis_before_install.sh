#!/bin/bash

docker pull shogun/shogun-dev
perl -pe 's/\$(\w+)/$ENV{$1}/g' configs/shogun-sdk/travis.env.in > travis.env
docker run -t -d -P --env-file travis.env \
	--name devenv -v $HOME/.ccache:/root/.ccache \
	-v $PWD:/opt/shogun shogun/shogun-dev /bin/sh \
	-c "mkdir /opt/shogun/build;bash"
