#ifdef HAVE_TFLOGGER

%include "std_vector.i"
%include "std_string.i"
%template(ParameterList) std::vector<std::string>;

%{
    #include <shogun/lib/parameter_observers/ParameterObserverInterface.h>
	#include <shogun/lib/parameter_observers/ParameterObserverTensorBoard.h>
    #include <shogun/lib/parameter_observers/ParameterObserverScalar.h>
    #include <shogun/lib/parameter_observers/ParameterObserverHistogram.h>
	#include <shogun/lib/parameter_observers/ParameterObserverCV.h>
	#include <shogun/lib/parameter_observers/ParameterObserverCVMKL.h>
	#include <shogun/lib/parameter_observers/ParameterObserverCVMulticlass.h>
%}

%include <shogun/lib/parameter_observers/ParameterObserverInterface.h>
%include <shogun/lib/parameter_observers/ParameterObserverTensorBoard.h>
%include <shogun/lib/parameter_observers/ParameterObserverScalar.h>
%include <shogun/lib/parameter_observers/ParameterObserverHistogram.h>
%include <shogun/lib/parameter_observers/ParameterObserverCV.h>
%include <shogun/lib/parameter_observers/ParameterObserverCVMKL.h>
%include <shogun/lib/parameter_observers/ParameterObserverCVMulticlass.h>

#endif // HAVE_TFLOGGER
