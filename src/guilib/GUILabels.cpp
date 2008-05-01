/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "lib/config.h"

#ifndef HAVE_SWIG
#include "guilib/GUILabels.h"
#include "features/Labels.h"
#include "lib/io.h"

#include <string.h>

CGUILabels::CGUILabels(CGUI * gui_)
: CSGObject(), gui(gui_), train_labels(NULL), test_labels(NULL)
{
}

CGUILabels::~CGUILabels()
{
	delete train_labels;
	delete test_labels;
}

bool CGUILabels::load(CHAR* filename, CHAR* target)
{
	CLabels* labels=NULL;

	if (strncmp(target, "TEST", 4)==0)
		labels=test_labels;
	else if (strncmp(target, "TRAIN", 5)==0)
		labels=train_labels;
	else
		SG_ERROR("Invalid target %s.\n", target);

	if (labels)
	{
		delete (labels);
		labels=new CLabels(filename);

		if (labels)
		{
			if (strncmp(target, "TEST", 4)==0)
				set_test_labels(labels);
			else
				set_train_labels(labels);

			return true;
		}
		else
			SG_ERROR("Loading labels failed.\n");
	}

	return false;
}

bool CGUILabels::save(CHAR* param)
{
	bool result=false;
	return result;
}
#endif
