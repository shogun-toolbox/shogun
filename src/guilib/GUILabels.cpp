#include "guilib/GUILabels.h"
#include "features/Labels.h"
#include "lib/io.h"

#include <string.h>

CGUILabels::CGUILabels(CGUI * gui_)
: gui(gui_), train_labels(NULL), test_labels(NULL)
{
}

CGUILabels::~CGUILabels()
{
	delete train_labels;
	delete test_labels;
}

bool CGUILabels::load(CHAR* param)
{
	param=CIO::skip_spaces(param);
	CHAR filename[1024]="";
	CHAR target[1024]="";
	bool result=false;

	if ((sscanf(param, "%s %s", filename, target))==2)
	{
		CLabels** f_ptr=NULL;

		if (strcmp(target,"TRAIN")==0)
		{
			f_ptr=&train_labels;
		}
		else if (strcmp(target,"TEST")==0)
		{
			f_ptr=&test_labels;
		}
		else
		{
			CIO::message(M_ERROR, "see help for parameters\n");
			return false;
		}

		if (f_ptr)
		{
			delete (*f_ptr);
			*f_ptr=new CLabels(filename);

			CLabels* label=*f_ptr;
			assert(label);
		}
	}
	else
		CIO::message(M_ERROR, "see help for params\n");

	return result;
}

bool CGUILabels::save(CHAR* param)
{
	bool result=false;
	return result;
}
