/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <string.h>

#include "base/SGObject.h"
#include "lib/DataType.h"

using namespace shogun;

TSGDataType::TSGDataType(EContainerType ctype, EPrimitveType ptype)
{
	m_ctype = ctype; m_ptype = ptype; m_length_y = m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 index_t* length)
{
	m_ctype = ctype; m_ptype = ptype; m_length_y = length;
	m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EPrimitveType ptype,
						 index_t* length_y, index_t* length_x)
{
	m_ctype = ctype; m_ptype = ptype; m_length_y = length_y;
	m_length_x = length_x;
}

void
TSGDataType::to_string(char* dest) const
{
	char* p = dest;

	switch (m_ctype) {
	case CT_SCALAR: strcpy(p, ""); break;
	case CT_VECTOR: strcpy(p, "Vector<"); break;
	case CT_MATRIX: strcpy(p, "Matrix<"); break;
	}

	switch (m_ptype) {
	case PT_BOOL: strcat(p, "bool"); break;
	case PT_CHAR: strcat(p, "char"); break;
	case PT_INT8: strcat(p, "int8"); break;
	case PT_UINT8: strcat(p, "uint8"); break;
	case PT_INT16: strcat(p, "int16"); break;
	case PT_UINT16: strcat(p, "uint16"); break;
	case PT_INT32: strcat(p, "int32"); break;
	case PT_UINT32: strcat(p, "uint32"); break;
	case PT_INT64: strcat(p, "int64"); break;
	case PT_UINT64: strcat(p, "uint64"); break;
	case PT_FLOAT32: strcat(p, "float32"); break;
	case PT_FLOAT64: strcat(p, "float64"); break;
	case PT_FLOATMAX: strcat(p, "floatmax"); break;
	case PT_SGSERIALIZABLE_PTR: strcat(p, "SGSerializable*"); break;
	}

	switch (m_ctype) {
	case CT_SCALAR: break;
	case CT_VECTOR: case CT_MATRIX:
		strcat(p, ">"); break;
	}
}

size_t
TSGDataType::sizeof_ptype(void) const
{
	switch (m_ptype) {
	case PT_BOOL: return sizeof (bool);
	case PT_CHAR: return sizeof (char);
	case PT_INT8: return sizeof (int8_t);
	case PT_UINT8: return sizeof (uint8_t);
	case PT_INT16: return sizeof (int16_t);
	case PT_UINT16: return sizeof (uint16_t);
	case PT_INT32: return sizeof (int32_t);
	case PT_UINT32: return sizeof (uint32_t);
	case PT_INT64: return sizeof (int64_t);
	case PT_UINT64: return sizeof (uint64_t);
	case PT_FLOAT32: return sizeof (float32_t);
	case PT_FLOAT64: return sizeof (float64_t);
	case PT_FLOATMAX: return sizeof (floatmax_t);
	case PT_SGSERIALIZABLE_PTR: return sizeof (CSGSerializable*);
	}

	return 0;
}
