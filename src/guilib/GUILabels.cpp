/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
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

			if (label)
				return true;
			else
				CIO::message(M_ERROR, "loading labels failed\n");
		}
	}
	else
		CIO::message(M_ERROR, "see help for params\n");

	return false;
}

bool CGUILabels::save(CHAR* param)
{
	bool result=false;
	return result;
}
#endif
