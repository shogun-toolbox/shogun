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

bool CGUILabels::load(char* param)
{
	param=CIO::skip_spaces(param);
	char filename[1024];
	char target[1024];
	bool result=false;
	bool allow_unknown=false;

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
			//allow_unknown=true;
		}
		else
		{
			CIO::message("see help for parameters\n");
			return false;
		}

		if (f_ptr)
		{
			delete (*f_ptr);
			*f_ptr=new CLabels(filename);

			CLabels* label=*f_ptr;
			assert(label);

			if (!allow_unknown)
			{
				bool invalids=false;

				for (long i=0; i<label->get_num_labels() && !invalids; i++)
				{

					if (label->get_label(i)==0)
						invalids=true;
				}

				if (invalids)
				{
					CIO::message("attempting to fix invalid labels (class 0), by setting them to class -1\n");
					for (long i=0; i<label->get_num_labels(); i++)
					{
						if (label->get_label(i)==0)
							label->set_label(i, -1);
					}
				}
			}
		}
	}
	else
		CIO::message("see help for params\n");

	return result;
}

bool CGUILabels::save(char* param)
{
	bool result=false;
	return result;
}
