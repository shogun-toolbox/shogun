#include "guilib/GUIKernel.h"
#include "kernel/Kernel.h"
#include "kernel/LinearKernel.h"
#include "lib/io.h"

#include <string.h>

CGUIKernel::CGUIKernel(CGUI * gui_): gui(gui_)
{
	kernel=NULL;
}

CGUIKernel::~CGUIKernel()
{
	delete kernel;
}

CKernel* CGUIKernel::get_kernel()
{
	return kernel;
}

bool CGUIKernel::set_kernel(char* param)
{
	param=CIO::skip_spaces(param);
	if (strcmp(param,"LINEAR")==0)
	{
		delete kernel;
		kernel=new CLinearKernel(false);  ////fixme make this an option
		return true;
	}
	else 
		CIO::not_implemented();

	return false;
}
