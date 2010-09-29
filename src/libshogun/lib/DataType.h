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

#include "lib/common.h"

namespace shogun
{
enum EContainerType {
	CT_SCALAR, CT_VECTOR, CT_MATRIX, CT_STRING, CT_SPARSE
};

enum EPrimitveType {
	PT_BOOL, PT_CHAR, PT_INT16, PT_UINT16, PT_INT32, PT_UINT32,
	PT_INT64, PT_UINT64, PT_FLOAT32, PT_FLOAT64, PT_FLOATMAX,
	PT_SGOBJECT_PTR
};

/* Datatypes that shogun supports. */
struct TSGDataType
{
	EContainerType m_ctype;
	EPrimitveType m_ptype;
	uint64_t *m_length_y, *m_length_x;

	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype);
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 uint64_t* length);
	explicit TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 uint64_t* length_y, uint64_t* length_x);

	inline bool operator==(const TSGDataType& a) {
		return m_ctype == a.m_ctype && m_ptype == a.m_ptype
			&& *m_length_y == *a.m_length_y
			&& *m_length_x == *a.m_length_x;
	}

	inline bool operator!=(const TSGDataType& a) {
		return !(*this == a);
	}

	void to_string(char* dest) const;
	size_t sizeof_ptype(void) const;
};
}
#endif // __DATATYPE_H__
