FROM debian:stretch-backports
MAINTAINER shogun@shogun-toolbox.org

RUN apt-get update -qq && apt-get upgrade -y && \
    apt-get install -qq --force-yes --no-install-recommends gnupg dirmngr
RUN apt-key adv --recv-keys --keyserver keyserver.ubuntu.com 8A9CA30DB3C431E3 60C317803A41BA51845E371A1E9377A2BA9EF27F
RUN printf "deb http://ppa.launchpad.net/timsc/swig-3.0.12/ubuntu xenial main\ndeb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu xenial main\n" | tee -a /etc/apt/sources.list

RUN apt-get update -qq && apt-get upgrade -y && \
    apt-get install -qq --force-yes --no-install-recommends make gcc-7 g++-7 swig \
    libc6-dev libbz2-dev ccache libarpack2-dev libatlas-base-dev \
    libblas-dev libglpk-dev libhdf5-serial-dev zlib1g-dev liblapacke-dev \
    libnlopt-dev liblpsolve55-dev libsnappy-dev liblzo2-dev \
    liblzma-dev libeigen3-dev python3-dev python3-numpy python3-matplotlib python3-scipy \
    python3-jinja2 python3-setuptools git-core wget jblas mono-devel mono-dmcs cli-common-dev \
    lua5.1 liblua5.1-0-dev octave liboctave-dev r-base-core clang-6.0 \
    openjdk-8-jdk ruby ruby-dev python3-ply sphinx-doc python3-pip \
    exuberant-ctags clang-format-3.8 libcolpack-dev rapidjson-dev lcov \
    protobuf-compiler libprotobuf-dev googletest
RUN apt-get -t stretch-backports install -qq --force-yes --no-install-recommends cmake

RUN pip3 install sphinx ply sphinxcontrib-bibtex sphinx_bootstrap_theme codecov
RUN gem install narray

ADD http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.19.tar.gz /tmp/
RUN cd /tmp && \
    tar zxpf arprec-2.2.19.tar.gz && \
    cd arprec && ./configure --enable-shared && \
    make install && ldconfig

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
