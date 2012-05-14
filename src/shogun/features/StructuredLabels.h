/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTUREDLABELS_H__
#define _STRUCTUREDLABELS_H__

#include <shogun/base/SGObject.h>
#include <shogun/lib/DynamicObjectArray.h>
#include <shogun/lib/StructuredData.h>

namespace shogun {

class CStructuredLabels : public CSGObject
{

	public:
		/** default constructor */
		CStructuredLabels();

		/** destructor */
		~CStructuredLabels();

		/** @return object name */
		inline virtual const char* get_name() const { return "StructuredLabels"; }

	private:
		/** the vector of labels */
		CDynamicObjectArray* labels;

}; /* class CStructuredLabels */

} /* namespace shogun */

#endif /* _STRUCTUREDLABELS_H__ */
