/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTUREDDATA_H__
#define _STRUCTUREDDATA_H__

#include <shogun/base/SGObject.h>

namespace shogun
{

class CStructuredData : public CSGObject
{

	public:
		/** default constructor */
		CStructuredData();

		/** destructor */
		virtual ~CStructuredData(); 

		/** @return object name */
		virtual const char* get_name() const { return "StructuredData"; }

}; 

} /* namespace shogun */

#endif /* _STRUCTUREDDATA_H__ */
