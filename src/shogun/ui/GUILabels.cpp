/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/ui/GUILabels.h>
#include <shogun/ui/SGInterface.h>

#include <shogun/lib/config.h>
#include <shogun/io/SGIO.h>
#include <shogun/features/Labels.h>

#include <string.h>

using namespace shogun;

CGUILabels::CGUILabels(CSGInterface* ui_)
: CSGObject(), ui(ui_), train_labels(NULL), test_labels(NULL)
{
}

CGUILabels::~CGUILabels()
{
	SG_UNREF(train_labels);
	SG_UNREF(test_labels);
}
