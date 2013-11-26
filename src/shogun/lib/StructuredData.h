/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Modified (M) 2013 Thoralf Klein
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTUREDDATA_H__
#define _STRUCTUREDDATA_H__

#include <shogun/base/SGRefObject.h>
// #include <shogun/base/SGObject.h>
#include <shogun/lib/StructuredDataTypes.h>

namespace shogun
{

/** @brief Base class of the components of StructuredLabels */
class StructuredData : public SGRefObject
{

	public:
		/** default constructor */
		StructuredData();

		/** destructor */
		virtual ~StructuredData();

		/** @return object name */
		virtual const char* get_name() const { return "StructuredData"; }

		/** get structured data type
		 *
		 * @return structured data type (a sequence, a tree, a graph, ...)
		 */
		virtual EStructuredDataType get_structured_data_type() const;

};

} /* namespace shogun */

#endif /* _STRUCTUREDDATA_H__ */
