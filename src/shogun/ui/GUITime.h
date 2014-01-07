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

#include <lib/config.h>
#include <lib/Time.h>
#include <base/SGObject.h>

namespace shogun
{
class CSGInterface;

/** @brief UI time */
class CGUITime : public CSGObject
{
	public:
		/** constructor */
		CGUITime() { };
		/** constructor
		 * @param interface
		 */
		CGUITime(CSGInterface* interface);
		/** destructor */
		~CGUITime();

		/** start */
		void start();
		/** stop */
		void stop();

		/** @return object name */
		virtual const char* get_name() const { return "GUITime"; }
	protected:
		/** ui */
		CSGInterface* ui;
		/** time */
		CTime* time;
};
}
#endif
