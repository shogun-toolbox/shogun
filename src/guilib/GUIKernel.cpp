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

bool CGUIKernel::load_kernel_init(char* param)
{
	bool result=false;
	if (kernel)
	{
		FILE* file=fopen(param, "r");
		if ((!file) || (!kernel->load_init(file)))
			CIO::message("reading from file %s failed!\n", param);
		else
		{
			CIO::message("successfully read kernel init data from \"%s\" !\n", param);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		CIO::message("no kernel set!\n");
	return result;
}

bool CGUIKernel::save_kernel_init(char* param)
{
	bool result=false;
	if (kernel)
	{
		FILE* file=fopen(param, "w");
		if ((!file) || (!kernel->save(file)))
			CIO::message("writing to file %s failed!\n", param);
		else
		{
			CIO::message("successfully written kernel init data into \"%s\" !\n", param);
			result=true;
		}

		if (file)
			fclose(file);
	}
	else
		CIO::message("no kernel set!\n");
	return result;
}
