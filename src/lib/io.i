%{
 #include "lib/common.h" 
 #include "lib/io.h" 
%}

%rename(IO) CIO;

%include "exception.i"
%include "lib/common.h" 
%include "lib/io.h" 

#ifdef USE_SWIG
%exception exception 
{
    $action
    SWIG_exception(SWIG_SystemError,const_cast<char*>(str));
}
#endif
