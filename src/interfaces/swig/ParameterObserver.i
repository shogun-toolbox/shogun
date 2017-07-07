%include "std_vector.i"
%include "std_string.i"
%template(ParameterList) std::vector<std::string>;

%{
    #include <shogun/lib/ParameterObserverInterface.h>
	#include <shogun/lib/ParameterObserverTensorBoard.h>
    #include <shogun/lib/ParameterObserverScalar.h>
    #include <shogun/lib/ParameterObserverHistogram.h>
%}

%include <shogun/lib/ParameterObserverInterface.h>
%include <shogun/lib/ParameterObserverTensorBoard.h>
%include <shogun/lib/ParameterObserverScalar.h>
%include <shogun/lib/ParameterObserverHistogram.h>
