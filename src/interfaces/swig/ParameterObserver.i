%rename(ParameterObserverCV) CParameterObserverCV;

%{
    #include <shogun/lib/observers/ParameterObserverInterface.h>
	#include <shogun/lib/observers/ParameterObserverCV.h>
#ifdef HAVE_TFLOGGER
	#include <shogun/lib/observers/ParameterObserverTensorBoard.h>
    #include <shogun/lib/observers/ParameterObserverScalar.h>
    #include <shogun/lib/observers/ParameterObserverHistogram.h>
#endif // HAVE_TFLOGGER
%}

%include <shogun/lib/observers/ParameterObserverInterface.h>
%include <shogun/lib/observers/ParameterObserverCV.h>
#ifdef HAVE_TFLOGGER
%include <shogun/lib/observers/ParameterObserverTensorBoard.h>
%include <shogun/lib/observers/ParameterObserverScalar.h>
%include <shogun/lib/observers/ParameterObserverHistogram.h>
#endif // HAVE_TFLOGGER
