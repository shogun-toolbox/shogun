%rename(ParameterObserverCV) CParameterObserverCV;

%{
    #include <shogun/lib/parameter_observers/ParameterObserverInterface.h>
	#include <shogun/lib/parameter_observers/ParameterObserverCV.h>
#ifdef HAVE_TFLOGGER
	#include <shogun/lib/parameter_observers/ParameterObserverTensorBoard.h>
    #include <shogun/lib/parameter_observers/ParameterObserverScalar.h>
    #include <shogun/lib/parameter_observers/ParameterObserverHistogram.h>
#endif // HAVE_TFLOGGER
%}

%include <shogun/lib/parameter_observers/ParameterObserverInterface.h>
%include <shogun/lib/parameter_observers/ParameterObserverCV.h>
#ifdef HAVE_TFLOGGER
%include <shogun/lib/parameter_observers/ParameterObserverTensorBoard.h>
%include <shogun/lib/parameter_observers/ParameterObserverScalar.h>
%include <shogun/lib/parameter_observers/ParameterObserverHistogram.h>
#endif // HAVE_TFLOGGER
