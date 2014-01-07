/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <lib/StructuredData.h>

using namespace shogun;

CStructuredData::CStructuredData()
: CSGObject()
{
}

CStructuredData::~CStructuredData()
{
}

EStructuredDataType CStructuredData::get_structured_data_type() const
{
	SG_ERROR("get_structured_data_type() not defined. \n"
		 "Make sure that STRUCTURED_DATA_TYPE(SDT) is defined "
		 "in every class that inherits from CStructuredData.\n");

	return SDT_UNKNOWN;
};
