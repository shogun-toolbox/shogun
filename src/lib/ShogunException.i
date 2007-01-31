%{
 #include "lib/ShogunException.h" 
%}

%exception 
{
    try
    {
        $action
    }
    catch (ShogunException e)
    {
        SWIG_exception(SWIG_SystemError,const_cast<char*>(e.get_exception_string()));
        SWIG_fail;
    }
}

%include "lib/ShogunException.h" 
%include "exception.i"
