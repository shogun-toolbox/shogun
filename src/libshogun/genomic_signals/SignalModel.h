/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Jonas Behr
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __MODEL_h__
#define __MODEL_h__

#include "lib/common.h"
#include "base/SGObject.h"

namespace shogun
{

/** @brief class SignalModel */
class CSignalModel : public CSGObject 
{
	public:

		/** constructor
		 */
		CSignalModel();

		virtual ~CSignalModel();

		/**
		 *
		 */

		/** 
		 * @return object name 
		 */
		inline virtual const char* get_name() const { return "SignalModel"; }
	protected:
};
}
#endif
