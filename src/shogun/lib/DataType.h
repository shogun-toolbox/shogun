/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando Jose Iglesias Garcia
 * Written (W) 2010,2012 Soeren Sonnenburg
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 2010 Berlin Institute of Technology
 * Copyright (C) 2012 Soeren Sonnenburg
 */

#ifndef __DATATYPE_H__
#define __DATATYPE_H__

#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>

#define PT_NOT_GENERIC	PT_SGOBJECT

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS
enum EContainerType
{
	CT_SCALAR=0,
	CT_VECTOR=1,
	CT_MATRIX=2,
	CT_NDARRAY=3,
	CT_SGVECTOR=4,
	CT_SGMATRIX=5,
	CT_UNDEFINED=6
};

enum EStructType
{
	ST_NONE=0,
	ST_STRING=1,
	ST_SPARSE=2,
	ST_UNDEFINED=3
};

enum EPrimitiveType
{
	PT_BOOL=0,
	PT_CHAR=1,
	PT_INT8=2,
	PT_UINT8=3,
	PT_INT16=4,
	PT_UINT16=5,
	PT_INT32=6,
	PT_UINT32=7,
	PT_INT64=8,
	PT_UINT64=9,
	PT_FLOAT32=10,
	PT_FLOAT64=11,
	PT_FLOATMAX=12,
	PT_SGOBJECT=13,
	PT_COMPLEX128=14,
	PT_UNDEFINED=15
};
#endif

/** @brief Datatypes that shogun supports. */
struct TSGDataType
{
	/** container type */
	EContainerType m_ctype;
	/** struct type */
	EStructType m_stype;
	/** primitive type */
	EPrimitiveType m_ptype;

	/** length y */
	index_t *m_length_y;
	/** length x */
	index_t *m_length_x;

	/** constructor
	 * @param ctype
	 * @param stype
	 * @param ptype
	 */
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype);
	/** constructor
	 * @param ctype
	 * @param stype
	 * @param ptype
	 * @param length
	 */
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length);
	/** constructor
	 * @param ctype
	 * @param stype
	 * @param ptype
	 * @param length_y
	 * @param length_x
	 */
	explicit TSGDataType(EContainerType ctype, EStructType stype,
						 EPrimitiveType ptype, index_t* length_y,
						 index_t* length_x);

	/** Compares the content of the data types, including the length fields if
	 * non-NULL
	 * @return other type to compare with
	 * @return true if equals, false otherwise
	 */
	bool equals(TSGDataType other);

	/** Compares the content of the data types, excluding the length fields
	 * @return other type to compare with
	 * @return true if equals, false otherwise
	 */
	bool equals_without_length(TSGDataType other);

	/** equality */
	bool operator==(const TSGDataType& a);
	/** inequality
	 * @param a
	 */
	inline bool operator!=(const TSGDataType& a)
	{
		return !(*this == a);
	}

	/** to string
	 * @param dest
	 * @param n
	 */
	void to_string(char* dest, size_t n) const;

	/** size of stype */
	size_t sizeof_stype() const;
	/** size of ptype */
	size_t sizeof_ptype() const;

	static size_t sizeof_ptype(EPrimitiveType ptype);
	static size_t sizeof_stype(EStructType stype, EPrimitiveType ptype);

	/** size of sparse entry
	 * @param ptype
	 */
	static size_t sizeof_sparseentry(EPrimitiveType ptype);

	/** offset of sparse entry
	 * @param ptype
	 */
	static size_t offset_sparseentry(EPrimitiveType ptype);

	/** stype to string
	 * @param dest
	 * @param stype
	 * @param ptype
	 * @param n
	 */
	static void stype_to_string(char* dest, EStructType stype,
	                            EPrimitiveType ptype, size_t n);
	/** ptype to string
	 * @param dest
	 * @param ptype
	 * @param n
	 */
	static void ptype_to_string(char* dest, EPrimitiveType ptype,
	                            size_t n);
	/** string to ptype
	 * @param ptype
	 * @param str
	 */
	static bool string_to_ptype(EPrimitiveType* ptype,
	                            const char* str);

	/** get size
	 * @return size of type in bytes
	 */
	size_t get_size();

	/** get num of elements
	 * @return number of (matrix, vector, scalar) elements of type
	 */
	int64_t get_num_elements();
};
}
#endif /* __DATATYPE_H__  */
