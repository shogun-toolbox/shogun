FROM ubuntu:14.04
MAINTAINER shogun@shogun-toolbox.org

RUN apt-get install -qq software-properties-common lsb-release
RUN add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) main universe restricted multiverse"
RUN apt-get update -qq
RUN apt-get upgrade -y

# install dependencies
RUN apt-get install -qq --no-install-recommends make gcc g++ libc6-dev libbz2-dev libarpack2-dev libatlas-base-dev libblas-dev libglpk-dev libhdf5-serial-dev zlib1g-dev liblapack-dev cmake libnlopt-dev liblpsolve55-dev libxml2-dev libsnappy-dev liblzo2-dev liblzma-dev swig2.0 python-dev python-numpy python-scipy python-matplotlib python-pandas python-setuptools python-networkx python-tornado python-pip python-zmq python-jinja2 git-core wget

RUN pip install ipython

# ADD http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.16.tar.gz /tmp/arprec-2.2.16.tar.gz
# use RUN instead to enable docker caching
RUN cd /tmp && wget 'http://crd.lbl.gov/~dhbailey/mpdist/arprec-2.2.16.tar.gz'
RUN cd /tmp;\
	tar zxpf arprec-2.2.16.tar.gz;\
	cd arprec-2.2.16;\
	./configure --enable-shared;\
	make install

# ADD http://cscapes.cs.purdue.edu/download/ColPack/ColPack-1.0.9.tar.gz /tmp/ColPack-1.0.9.tar.gz
RUN cd /tmp && wget 'http://cscapes.cs.purdue.edu/download/ColPack/ColPack-1.0.9.tar.gz'
RUN cd /tmp;\
	tar zxpf ColPack-1.0.9.tar.gz;\
	cd ColPack-1.0.9;\
	./configure;\
	make install

RUN git clone https://github.com/shogun-toolbox/shogun-data.git
ADD shogun-data /data
ADD doc/ipython-notebooks /tmp/notebooks

RUN mkdir build;\
	cd build;\
	cmake -DPythonModular=ON -DUSE_SWIG_DIRECTORS=ON -DBUNDLE_EIGEN=ON -DBUNDLE_JSON=ON ..;\
	make;\
	make install

EXPOSE 8888

WORKDIR /tmp/notebooks
CMD ['ipython', 'notebook', '--pylab=inline', '--ip=*']
