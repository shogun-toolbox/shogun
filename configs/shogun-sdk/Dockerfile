FROM debian:buster-backports
MAINTAINER shogun@shogun-toolbox.org

RUN apt-get update -qq && apt-get upgrade -y && \
    apt-get install -qq --force-yes --no-install-recommends make gcc g++ \
    libc6-dev libbz2-dev ccache libarpack2-dev libatlas-base-dev \
    libblas-dev libglpk-dev libhdf5-serial-dev zlib1g-dev liblapacke-dev \
    libnlopt-dev liblpsolve55-dev libsnappy-dev liblzo2-dev \
    liblzma-dev libeigen3-dev python3-dev python3-numpy python3-matplotlib python3-scipy \
    python3-jinja2 python3-setuptools git-core wget jblas mono-devel cli-common-dev \
    octave liboctave-dev r-base-core clang \
    openjdk-11-jdk default-jre-headless ruby ruby-dev python3-ply sphinx-doc \
    python3-pip exuberant-ctags clang-format libcolpack-dev rapidjson-dev lcov \
    protobuf-compiler libprotobuf-dev googletest gnupg dirmngr cmake ninja-build \
    libspdlog-dev bison libpcre3-dev

RUN pip3 install sphinx ply sphinxcontrib-bibtex sphinx_bootstrap_theme codecov
RUN gem install narray

ADD https://github.com/ReactiveX/RxCpp/archive/4.1.0.tar.gz /tmp/
RUN cd /tmp;\
    tar -xvf 4.1.0.tar.gz;\
    cd RxCpp-4.1.0/projects/;\
    mkdir build;\
    cd build;\
    cmake ../../;\
    make install;

ADD https://github.com/shogun-toolbox/tflogger/archive/v0.1.1.tar.gz /tmp/
RUN cd /tmp;\
    tar -xvf v0.1.1.tar.gz;\
    cd tflogger-0.1.1;\
    mkdir build;\
    cd build;\
    cmake ../;\
    make install;

ADD https://github.com/swig/swig/archive/rel-4.0.1.tar.gz /tmp/
RUN cd /tmp;\
    tar -xvf rel-4.0.1.tar.gz;\
    cd swig-rel-4.0.1;\
    ./autogen.sh;\
    ./configure;\
    make && make install && ldconfig
