%module test

%{
#define SWIG_FILE_WITH_INIT
#include "test_swig.h"
%}

%include <std_shared_ptr.i>
%include <std_string.i>
%shared_ptr(test::Base)
%shared_ptr(test::Machine);

%include "test_swig.h"
namespace test {
%template(set) Base::set<int, int>;

%define SET(sg_class)
%template(set) Base::set<sg_class, sg_class, void>;
%enddef

SET(Base)
}