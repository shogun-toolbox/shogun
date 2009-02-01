%{
 #include "lib/ShogunException.h"
%}

%exception
{
	try
	{
		$action
	}
	catch (std::bad_alloc)
	{
		SWIG_exception(SWIG_MemoryError, const_cast<char*>("Out of memory error.\n"));
		SWIG_fail;
	}
	catch (ShogunException e)
	{
		SWIG_exception(SWIG_SystemError, const_cast<char*>(e.get_exception_string()));
		SWIG_fail;
	}
}

%include "lib/ShogunException.h"
%include "exception.i"
