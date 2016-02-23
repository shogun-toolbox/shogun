FROM ubuntu:14.04
MAINTAINER shogun@shogun-toolbox.org

RUN apt-get install -qq software-properties-common lsb-release
RUN add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
RUN add-apt-repository -y ppa:rosmo/swig3.0.7
RUN add-apt-repository -y ppa:webupd8team/java
RUN add-apt-repository -y ppa:george-edison55/cmake-3.x
RUN apt-get update -qq
RUN apt-get upgrade -y

RUN echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
RUN echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections

# install dependencies
RUN apt-get install -qq --force-yes --no-install-recommends make gcc g++ \
    libc6-dev libbz2-dev libjson-c-dev ccache libarpack2-dev libatlas-base-dev \
    libblas-dev libglpk-dev libhdf5-serial-dev zlib1g-dev liblapack-dev cmake \
    libnlopt-dev liblpsolve55-dev libxml2-dev libsnappy-dev liblzo2-dev \
    liblzma-dev libeigen3-dev swig3.0 python-dev python-numpy python-scipy \
    python-jinja2 git-core wget jblas mono-devel mono-gmcs cli-common-dev \
    lua5.1 liblua5.1-0-dev octave liboctave-dev r-base-core clang \
    oracle-java8-installer ruby ruby-dev python-ply sphinx-doc

RUN pip install sphinxcontrib-bibtex

RUN gem install narray
RUN cd /usr/bin && ln -s swig3.0 swig && ln -s ccache-swig3.0 ccache-swig
ADD http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.18.tar.gz /tmp
RUN cd /tmp;\
    tar zxpf arprec-2.2.18.tar.gz;\
    cd arprec-2.2.18;\
    ./configure --enable-shared;\
    make install; ldconfig
