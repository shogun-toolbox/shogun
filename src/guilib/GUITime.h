/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2007 Soeren Sonnenburg
 * Copyright (C) 1999-2007 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUITIME__H_
#define __GUITIME__H_

#include "lib/config.h"

#ifndef HAVE_SWIG

#include "base/SGObject.h"
#include "lib/Time.h"

class CGUI;

class CGUITime : public CSGObject
{
	public:
		CGUITime(CGUI *);
		~CGUITime();

		void start();
		void stop();

	protected:
		CGUI* gui;
		CTime* time;
};
#endif //HAVE_SWIG
#endif
