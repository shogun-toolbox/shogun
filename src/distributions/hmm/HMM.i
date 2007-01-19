%module(directors="1") HMM
%{
#include "distributions/Distribution.h"
#include "distributions/hmm/HMM.h"
%}

%include "lib/common.i"

/* create code for all classes with virtual methods */
%feature("director");

%include "base/SGObject.i"
%include "distributions/Distribution.h"
%include "distributions/hmm/HMM.h"


