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
	int size=100;
	char type[1024];
	param=CIO::skip_spaces(param);

	if (sscanf(param, "%s %d", type, &size) >= 1)
	{
		if (strcmp(type,"LINEAR")==0)
		{
			int scale=1;
			sscanf(param, "%s %d %d", type, &size, &scale);
			delete kernel;
			kernel=new CLinearKernel(size, scale==1);
			return true;
		}
		else 
			CIO::not_implemented();
	}
	else 
		CIO::message("see help for params!\n");
	return false;
}
