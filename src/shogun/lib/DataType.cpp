/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include <string.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGSparseVector.h>

using namespace shogun;

TSGDataType::TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype)
{
	m_ctype = ctype, m_stype = stype, m_ptype = ptype;
	m_length_y = m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length)
{
	m_ctype = ctype, m_stype = stype, m_ptype = ptype;
	m_length_y = length, m_length_x = NULL;
}

TSGDataType::TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length_y,
						 index_t* length_x)
{
	m_ctype = ctype, m_stype = stype, m_ptype = ptype;
	m_length_y = length_y, m_length_x = length_x;
}

bool
TSGDataType::operator==(const TSGDataType& a)
{
	/* handle CT_SG* and SG_* ambiguity */
	bool ctype_equal=false;
	if ((m_ctype==CT_VECTOR && a.m_ctype==CT_SGVECTOR) ||
			(m_ctype==CT_SGVECTOR && a.m_ctype==CT_VECTOR) ||
			(m_ctype==CT_MATRIX && a.m_ctype==CT_SGMATRIX) ||
			(m_ctype==CT_SGMATRIX && a.m_ctype==CT_MATRIX) ||
			(m_ctype==a.m_ctype))
		ctype_equal=true;

	bool result = ctype_equal && m_stype == a.m_stype
		&& m_ptype == a.m_ptype;

	result &= m_length_y != NULL && a.m_length_y != NULL
		? *m_length_y == *a.m_length_y: m_length_y == a.m_length_y;
	result &= m_length_x != NULL && a.m_length_x != NULL
		? *m_length_x == *a.m_length_x: m_length_x == a.m_length_x;

	return result;
}

bool TSGDataType::equals_without_length(TSGDataType other)
{
	if (m_ctype!=other.m_ctype)
	{
		SG_SDEBUG("leaving TSGDataType::equals_wihtout_length(): container types are "
				"different\n");
		return false;
	}

	if (m_stype!=other.m_stype)
	{
		SG_SDEBUG("leaving TSGDataType::equals_wihtout_length(): struct types are "
				"different\n");
		return false;
	}

	if (m_ptype!=other.m_ptype)
	{
		SG_SDEBUG("leaving TSGDataType::equals_wihtout_length(): primitive types are "
				"different\n");
		return false;
	}

	SG_SDEBUG("leaving TSGDataType::equals_wihtout_length(): data types "
			"without lengths are equal\n");
	return true;
}

bool TSGDataType::equals(TSGDataType other)
{
	SG_SDEBUG("entering TSGDataType::equals()\n");

	if (!equals_without_length(other))
	{
		SG_SDEBUG("leaving TSGDataType::equals(): Data types without lengths "
				"are not equal\n");
		return false;
	}

	if ((!m_length_y && other.m_length_y) || (m_length_y && !other.m_length_y))
	{
		SG_SDEBUG("leaving TSGDataType::equals(): length_y is at %p while "
				"other's length_y is at %p\n", m_length_y, other.m_length_y);
		return false;
	}

	if (m_length_y && other.m_length_y)
	{
		if (*m_length_y!=*other.m_length_y)
		{
			SG_SDEBUG("leaving TSGDataType::equals(): length_y=%d while "
					"other's length_y=%d\n", *m_length_y, *other.m_length_y);
			return false;
		}
	}

	if ((!m_length_x && other.m_length_x) || (m_length_x && !other.m_length_x))
	{
		SG_SDEBUG("leaving TSGDataType::equals(): m_length_x is at %p while "
				"other's m_length_x is at %p\n", m_length_x, other.m_length_x);
		return false;
	}

	if (m_length_x && other.m_length_x)
	{
		if (*m_length_x!=*other.m_length_x)
		{
			SG_SDEBUG("leaving TSGDataType::equals(): m_length_x=%d while "
					"other's m_length_x=%d\n", *m_length_x, *other.m_length_x);
			return false;
		}
	}

	SG_SDEBUG("leaving TSGDataType::equals(): datatypes are equal\n");
	return true;
}

void
TSGDataType::to_string(char* dest, size_t n) const
{
	char* p = dest;

	switch (m_ctype) {
	case CT_SCALAR: strncpy(p, "", n); break;
	case CT_VECTOR: strncpy(p, "Vector<", n); break;
	case CT_SGVECTOR: strncpy(p, "SGVector<", n); break;
	case CT_MATRIX: strncpy(p, "Matrix<", n); break;
	case CT_SGMATRIX: strncpy(p, "SGMatrix<", n); break;
	case CT_NDARRAY: strncpy(p, "N-Dimensional Array<", n); break;
	case CT_UNDEFINED: default: strncpy(p, "Undefined", n); break;
	}

	if (m_ctype != CT_UNDEFINED)
	{
		size_t np = strlen(p);
		stype_to_string(p + np, m_stype, m_ptype, n - np - 2);
	}

	switch (m_ctype) {
	case CT_SCALAR: break;
	case CT_VECTOR:
	case CT_SGVECTOR:
	case CT_MATRIX:
	case CT_SGMATRIX:
	case CT_NDARRAY: strcat(p, ">"); break;
	case CT_UNDEFINED: default: break;
	}
}

size_t
TSGDataType::sizeof_stype() const
{
	return sizeof_stype(m_stype, m_ptype);
}

size_t
TSGDataType::sizeof_ptype() const
{
	return sizeof_ptype(m_ptype);
}

size_t
TSGDataType::sizeof_stype(EStructType stype, EPrimitiveType ptype)
{
	switch (stype) {
	case ST_NONE: return sizeof_ptype(ptype);
	case ST_STRING:
		switch (ptype) {
		case PT_BOOL: return sizeof (SGString<bool>);
		case PT_CHAR: return sizeof (SGString<char>);
		case PT_INT8: return sizeof (SGString<int8_t>);
		case PT_UINT8: return sizeof (SGString<uint8_t>);
		case PT_INT16: return sizeof (SGString<int16_t>);
		case PT_UINT16: return sizeof (SGString<uint16_t>);
		case PT_INT32: return sizeof (SGString<int32_t>);
		case PT_UINT32: return sizeof (SGString<uint32_t>);
		case PT_INT64: return sizeof (SGString<int64_t>);
		case PT_UINT64: return sizeof (SGString<uint64_t>);
		case PT_FLOAT32: return sizeof (SGString<float32_t>);
		case PT_FLOAT64: return sizeof (SGString<float64_t>);
		case PT_FLOATMAX: return sizeof (SGString<floatmax_t>);
		case PT_COMPLEX128:
			SG_SWARNING("TGSDataType::sizeof_stype(): Strings are"
				" not supported for complex128_t\n");
			return -1;
		case PT_SGOBJECT:
			SG_SWARNING("TGSDataType::sizeof_stype(): Strings are"
				" not supported for SGObject\n");
			return -1;
		case PT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		break;
	case ST_SPARSE:
		switch (ptype) {
		case PT_BOOL: return sizeof (SGSparseVector<bool>);
		case PT_CHAR: return sizeof (SGSparseVector<char>);
		case PT_INT8: return sizeof (SGSparseVector<int8_t>);
		case PT_UINT8: return sizeof (SGSparseVector<uint8_t>);
		case PT_INT16: return sizeof (SGSparseVector<int16_t>);
		case PT_UINT16: return sizeof (SGSparseVector<uint16_t>);
		case PT_INT32: return sizeof (SGSparseVector<int32_t>);
		case PT_UINT32: return sizeof (SGSparseVector<uint32_t>);
		case PT_INT64: return sizeof (SGSparseVector<int64_t>);
		case PT_UINT64: return sizeof (SGSparseVector<uint64_t>);
		case PT_FLOAT32: return sizeof (SGSparseVector<float32_t>);
		case PT_FLOAT64: return sizeof (SGSparseVector<float64_t>);
		case PT_FLOATMAX: return sizeof (SGSparseVector<floatmax_t>);
		case PT_COMPLEX128: return sizeof (SGSparseVector<complex128_t>);
		case PT_SGOBJECT: return -1;
		case PT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined primitive type\n");
			break;
		}
		break;
	case ST_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}

	return -1;
}

size_t
TSGDataType::sizeof_ptype(EPrimitiveType ptype)
{
	switch (ptype) {
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
	case PT_COMPLEX128: return sizeof (complex128_t);
	case PT_SGOBJECT: return sizeof (CSGObject*);
	case PT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined primitive type\n");
		break;
	}

	return -1;
}

size_t
TSGDataType::sizeof_sparseentry(EPrimitiveType ptype)
{
	switch (ptype) {
	case PT_BOOL: return sizeof (SGSparseVectorEntry<bool>);
	case PT_CHAR: return sizeof (SGSparseVectorEntry<char>);
	case PT_INT8: return sizeof (SGSparseVectorEntry<int8_t>);
	case PT_UINT8: return sizeof (SGSparseVectorEntry<uint8_t>);
	case PT_INT16: return sizeof (SGSparseVectorEntry<int16_t>);
	case PT_UINT16: return sizeof (SGSparseVectorEntry<uint16_t>);
	case PT_INT32: return sizeof (SGSparseVectorEntry<int32_t>);
	case PT_UINT32: return sizeof (SGSparseVectorEntry<uint32_t>);
	case PT_INT64: return sizeof (SGSparseVectorEntry<int64_t>);
	case PT_UINT64: return sizeof (SGSparseVectorEntry<uint64_t>);
	case PT_FLOAT32: return sizeof (SGSparseVectorEntry<float32_t>);
	case PT_FLOAT64: return sizeof (SGSparseVectorEntry<float64_t>);
	case PT_FLOATMAX: return sizeof (SGSparseVectorEntry<floatmax_t>);
	case PT_COMPLEX128: return sizeof (SGSparseVectorEntry<complex128_t>);
	case PT_SGOBJECT: return -1;
	case PT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined primitive type\n");
		break;
	}

	return -1;
}

#define ENTRY_OFFSET(k, type)									\
	((char*) &((SGSparseVectorEntry<type>*) (k))->entry - (char*) (k))
size_t
TSGDataType::offset_sparseentry(EPrimitiveType ptype)
{
	size_t result = -1; void* x = &result;

	switch (ptype) {
	case PT_BOOL: result = ENTRY_OFFSET(x, bool); break;
	case PT_CHAR: result = ENTRY_OFFSET(x, char); break;
	case PT_INT8: result = ENTRY_OFFSET(x, int8_t); break;
	case PT_UINT8: result = ENTRY_OFFSET(x, uint8_t); break;
	case PT_INT16: result = ENTRY_OFFSET(x, int16_t); break;
	case PT_UINT16: result = ENTRY_OFFSET(x, uint16_t); break;
	case PT_INT32: result = ENTRY_OFFSET(x, int32_t); break;
	case PT_UINT32: result = ENTRY_OFFSET(x, uint32_t); break;
	case PT_INT64: result = ENTRY_OFFSET(x, int64_t); break;
	case PT_UINT64: result = ENTRY_OFFSET(x, uint64_t); break;
	case PT_FLOAT32: result = ENTRY_OFFSET(x, float32_t); break;
	case PT_FLOAT64: result = ENTRY_OFFSET(x, float64_t); break;
	case PT_FLOATMAX: result = ENTRY_OFFSET(x, floatmax_t); break;
	case PT_COMPLEX128: result = ENTRY_OFFSET(x, complex128_t); break;
	case PT_SGOBJECT: return -1;
	case PT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined primitive type\n");
		break;
	}

	return result;
}

void
TSGDataType::stype_to_string(char* dest, EStructType stype,
							 EPrimitiveType ptype, size_t n)
{
	char* p = dest;

	switch (stype) {
	case ST_NONE: strncpy(p, "", n); break;
	case ST_STRING: strncpy(p, "String<", n); break;
	case ST_SPARSE: strncpy(p, "Sparse<", n); break;
	case ST_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}

	size_t np = strlen(p);
	ptype_to_string(p + np, ptype, n - np - 2);

	switch (stype) {
	case ST_NONE: break;
	case ST_STRING: case ST_SPARSE:
		strcat(p, ">"); break;
	case ST_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined structure type\n");
		break;
	}
}

void
TSGDataType::ptype_to_string(char* dest, EPrimitiveType ptype,
							 size_t n)
{
	char* p = dest;

	switch (ptype) {
	case PT_BOOL: strncpy(p, "bool", n); break;
	case PT_CHAR: strncpy(p, "char", n); break;
	case PT_INT8: strncpy(p, "int8", n); break;
	case PT_UINT8: strncpy(p, "uint8", n); break;
	case PT_INT16: strncpy(p, "int16", n); break;
	case PT_UINT16: strncpy(p, "uint16", n); break;
	case PT_INT32: strncpy(p, "int32", n); break;
	case PT_UINT32: strncpy(p, "uint32", n); break;
	case PT_INT64: strncpy(p, "int64", n); break;
	case PT_UINT64: strncpy(p, "uint64", n); break;
	case PT_FLOAT32: strncpy(p, "float32", n); break;
	case PT_FLOAT64: strncpy(p, "float64", n); break;
	case PT_FLOATMAX: strncpy(p, "floatmax", n); break;
	case PT_COMPLEX128: strncpy(p, "complex128", n); break;
	case PT_SGOBJECT: strncpy(p, "SGSerializable*", n); break;
	case PT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined primitive type\n");
		break;
	}
}

bool
TSGDataType::string_to_ptype(EPrimitiveType* ptype, const char* str)
{
	if (strcmp(str, "bool") == 0) {
		*ptype = PT_BOOL; return true; }
	if (strcmp(str, "char") == 0) {
		*ptype = PT_CHAR; return true; }
	if (strcmp(str, "int8") == 0) {
		*ptype = PT_INT8; return true; }
	if (strcmp(str, "uint8") == 0) {
		*ptype = PT_UINT8; return true; }
	if (strcmp(str, "int16") == 0) {
		*ptype = PT_INT16; return true; }
	if (strcmp(str, "uint16") == 0) {
		*ptype = PT_UINT16; return true; }
	if (strcmp(str, "int32") == 0) {
		*ptype = PT_INT32; return true; }
	if (strcmp(str, "uint32") == 0) {
		*ptype = PT_UINT32; return true; }
	if (strcmp(str, "int64") == 0) {
		*ptype = PT_INT64; return true; }
	if (strcmp(str, "uint64") == 0) {
		*ptype = PT_UINT64; return true; }
	if (strcmp(str, "float32") == 0) {
		*ptype = PT_FLOAT32; return true; }
	if (strcmp(str, "float64") == 0) {
		*ptype = PT_FLOAT64; return true; }
	if (strcmp(str, "floatmax") == 0) {
		*ptype = PT_FLOATMAX; return true; }
	if (strcmp(str, "complex128") == 0) {
		*ptype = PT_COMPLEX128; return true; }
	if (strcmp(str, "SGSerializable*") == 0) {
		*ptype = PT_SGOBJECT; return true; }

	/* Make sure that the compiler will warn at this position.  */
	switch (*ptype) {
	case PT_BOOL: case PT_CHAR: case PT_INT8: case PT_UINT8:
	case PT_INT16: case PT_UINT16: case PT_INT32: case PT_UINT32:
	case PT_INT64: case PT_UINT64: case PT_FLOAT32: case PT_FLOAT64:
	case PT_FLOATMAX: case PT_COMPLEX128: case PT_SGOBJECT: break;
	case PT_UNDEFINED: default:
		SG_SERROR("Implementation error: undefined primitive type\n");
		break;
	}

	return false;
}

size_t TSGDataType::get_size()
{
	switch (m_stype)
	{
		case ST_NONE:
			return get_num_elements()*sizeof_ptype();
		case ST_STRING:
			if (m_ptype==PT_SGOBJECT)
				return 0;

			return get_num_elements()*sizeof_stype();
		case ST_SPARSE:
			if (m_ptype==PT_SGOBJECT)
				return 0;

			return get_num_elements()*sizeof_sparseentry(m_ptype);
		case ST_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined structure type\n");
			break;
	}

	return 0;
}

int64_t TSGDataType::get_num_elements()
{
	switch (m_ctype)
	{
		case CT_SCALAR:
			return 1;
		case CT_VECTOR: case CT_SGVECTOR:
			/* length_y contains the length for vectors */
			return *m_length_y;
		case CT_MATRIX: case CT_SGMATRIX:
			return (*m_length_y)*(*m_length_x);
		case CT_NDARRAY:
			SG_SNOTIMPLEMENTED
		case CT_UNDEFINED: default:
			SG_SERROR("Implementation error: undefined container type\n");
			break;
	}
	return 0;
}
