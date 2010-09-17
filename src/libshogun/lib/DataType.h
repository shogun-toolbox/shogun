/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __DATATYPE_H__
#define __DATATYPE_H__
namespace shogun
{
enum TContainerType {
	CT_SCALAR
};

enum TPrimitveType {
	PT_BOOL, PT_CHAR, PT_INT16, PT_INT32, PT_INT64, PT_FLOAT32,
	PT_FLOAT64, PT_FLOATMAX, PT_SGOBJECT_PTR
};

/* Datatypes that shogun supports. */
struct TSGDataType
{
	explicit TSGDataType(TContainerType ctype, TPrimitveType ptype) {
		m_ctype = ctype; m_ptype = ptype;
	}

	TContainerType m_ctype;
	TPrimitveType m_ptype;
};
}
#endif // __DATATYPE_H__
