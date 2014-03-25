/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#ifndef _STRUCTURED_DATA_TYPES__H__
#define _STRUCTURED_DATA_TYPES__H__

#include <shogun/lib/config.h>

namespace shogun {

/** structured data type */
enum EStructuredDataType
{
	SDT_UNKNOWN = 0,
	SDT_REAL = 1,
	SDT_SEQUENCE = 2,
	SDT_FACTOR_GRAPH = 3
};

#define STRUCTURED_DATA_TYPE(SDT) \
	/** returns structured data type used by the labels \
	 *
	 * @return structured data type
	 */ \
	virtual EStructuredDataType get_structured_data_type() const { return SDT; }
}

#endif /* _STRUCTURED_DATA_TYPES__H__ */
