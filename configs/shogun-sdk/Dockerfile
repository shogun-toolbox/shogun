FROM ubuntu:16.04
MAINTAINER shogun@shogun-toolbox.org

RUN apt-get update && apt-get install -qq software-properties-common lsb-release
RUN add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) multiverse"
RUN add-apt-repository -y ppa:webupd8team/java
RUN apt-get update -qq
RUN apt-get upgrade -y

RUN echo debconf shared/accepted-oracle-license-v1-1 select true | debconf-set-selections
RUN echo debconf shared/accepted-oracle-license-v1-1 seen true | debconf-set-selections

# install dependencies
RUN apt-get install -qq --force-yes --no-install-recommends make gcc g++ \
    libc6-dev libbz2-dev libjson-c-dev ccache libarpack2-dev libatlas-base-dev \
    libblas-dev libglpk-dev libhdf5-serial-dev zlib1g-dev liblapack-dev cmake \
    libnlopt-dev liblpsolve55-dev libxml2-dev libsnappy-dev liblzo2-dev \
    liblzma-dev libeigen3-dev swig3.0 python-dev python-numpy python-matplotlib python-scipy \
    python-jinja2 python-setuptools git-core wget jblas mono-devel mono-dmcs cli-common-dev \
    lua5.1 liblua5.1-0-dev octave liboctave-dev r-base-core clang \
    oracle-java8-installer ruby ruby-dev python-ply sphinx-doc python-pip \
    exuberant-ctags clang-format-3.8 libcereal-dev libcolpack-dev lcov

RUN pip install sphinx ply sphinxcontrib-bibtex sphinx_bootstrap_theme codecov

RUN gem install narray
RUN cd /usr/bin && ln -s swig3.0 swig && ln -s ccache-swig3.0 ccache-swig
ADD http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.18.tar.gz /tmp/
RUN cd /tmp;\
    tar zxpf arprec-2.2.18.tar.gz;\
    cd arprec-2.2.18;\
    ./configure --enable-shared;\
    make install; ldconfig

ADD http://www.scala-lang.org/files/archive/scala-2.11.7.deb /tmp/scala.deb
RUN dpkg -i /tmp/scala.deb

ADD http://dl.bintray.com/sbt/debian/sbt-0.13.6.deb /tmp/sbt.deb
RUN dpkg -i /tmp/sbt.deb

ADD https://github.com/Reactive-Extensions/RxCpp/archive/v4.0.0.tar.gz /tmp/
RUN cd /tmp;\
    tar -xvf v4.0.0.tar.gz;\
    cd RxCpp-4.0.0/projects/;\
    mkdir build;\
    cd build;\
    cmake ../../;\
    make install;

ADD https://github.com/shogun-toolbox/tflogger/archive/v0.1.0.tar.gz /tmp/
RUN cd /tmp;\
    tar -xvf v0.1.0.tar.gz;\
    cd tflogger-0.1.0;\
    mkdir build;\
    cd build;\
    cmake ../;\
    make install;
