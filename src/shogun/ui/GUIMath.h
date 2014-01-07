/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __GUIMATH__H__
#define __GUIMATH__H__

#include <lib/config.h>
#include <base/SGObject.h>

namespace shogun
{
class CSGInterface;

/** @brief UI math */
class CGUIMath : public CSGObject
{
	public:
		/** constructor */
		CGUIMath() {};
		/** constructor
		 * @param interface
		 */
		CGUIMath(CSGInterface* interface);
		/** set threshold
		 * @param value
		 */
		void set_threshold(float64_t value);
		/** init random
		 * @param initseed
		 */
		void init_random(uint32_t initseed=0);

		/** @return object name */
		virtual const char* get_name() const { return "GUIMath"; }
	protected:
		/** ui */
		CSGInterface* ui;
		/** threshold */
		float64_t threshold;
};
}
#endif
