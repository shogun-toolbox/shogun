/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Soumyajit De, Thoralf Klein, 
 *          Bjoern Esser
 */

#include <string.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/io/SGIO.h>

using namespace shogun;

namespace shogun
{
	std::string ptype_name(EPrimitiveType pt)
	{
		switch (pt)
		{
		case PT_BOOL:
			return "BOOL";
		case PT_CHAR:
			return "CHAR";
		case PT_INT8:
			return "INT8";
		case PT_UINT8:
			return "UINT8";
		case PT_INT16:
			return "INT16";
		case PT_UINT16:
			return "INT16";
		case PT_INT32:
			return "INT32";
		case PT_UINT32:
			return "UINT32";
		case PT_INT64:
			return "INT64";
		case PT_UINT64:
			return "UINT64";
		case PT_FLOAT32:
			return "FLOAT32";
		case PT_FLOAT64:
			return "FLOAT64";
		case PT_FLOATMAX:
			return "FLOATMAX";
		case PT_SGOBJECT:
			return "SGOBJECT";
		case PT_COMPLEX128:
			return "COMPLEX128";
		default:
			not_implemented(SOURCE_LOCATION);
			return "UNKNOWN";
		}
	}
}

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
			error("Implementation error: undefined primitive type");
			break;
		}
		break;
	case ST_UNDEFINED: default:
		error("Implementation error: undefined structure type");
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
	case PT_SGOBJECT: return sizeof (SGObject*);
	case PT_UNDEFINED: default:
		error("Implementation error: undefined primitive type");
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
		error("Implementation error: undefined primitive type");
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
		error("Implementation error: undefined primitive type");
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
	case ST_SPARSE: strncpy(p, "Sparse<", n); break;
	case ST_UNDEFINED: default:
		error("Implementation error: undefined structure type");
		break;
	}

	size_t np = strlen(p);
	ptype_to_string(p + np, ptype, n - np - 2);

	switch (stype) {
	case ST_NONE: break;
	case ST_SPARSE:
		strcat(p, ">"); break;
	case ST_UNDEFINED: default:
		error("Implementation error: undefined structure type");
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
		error("Implementation error: undefined primitive type");
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
		error("Implementation error: undefined primitive type");
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
		case ST_SPARSE:
			if (m_ptype==PT_SGOBJECT)
				return 0;

			return get_num_elements()*sizeof_sparseentry(m_ptype);
		case ST_UNDEFINED: default:
			error("Implementation error: undefined structure type");
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
			not_implemented(SOURCE_LOCATION);
		case CT_UNDEFINED: default:
			error("Implementation error: undefined container type");
			break;
	}
	return 0;
}
