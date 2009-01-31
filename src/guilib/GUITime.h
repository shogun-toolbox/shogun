/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUITIME__H_
#define __GUITIME__H_

#include "lib/config.h"

#ifndef HAVE_SWIG

#include "base/SGObject.h"
#include "lib/Time.h"

class CSGInterface;

class CGUITime : public CSGObject
{
	public:
		CGUITime(CSGInterface* interface);
		~CGUITime();

		void start();
		void stop();

		/** @return object name */
		inline virtual const char* get_name() { return "GUITime"; }
	protected:
		CSGInterface* ui;
		CTime* time;
};
#endif //HAVE_SWIG
#endif
