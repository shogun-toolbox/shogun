/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2006 Soeren Sonnenburg
 * Written (W) 1999-2006 Gunnar Raetsch
 * Written (W) 1999-2006 Fabio De Bona
 * Copyright (C) 1999-2006 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include "guilib/GUITime.h"

CGUITime::CGUITime(CGUI* g) : gui(g)
{
	time=new CTime();
	ASSERT(time);
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
