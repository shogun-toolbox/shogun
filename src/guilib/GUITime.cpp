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
#include "guilib/GUITime.h"
#include "interface/SGInterface.h"

CGUITime::CGUITime(CSGInterface* ui_) : CSGObject(), ui(ui_)
{
	time=new CTime();
}

CGUITime::~CGUITime()
{
	delete time;
}

void CGUITime::start()
{
	time->start();
}

void CGUITime::stop()
{
	time->stop();
	time->time_diff_sec(true);
}
#endif
