/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __FEATURE_STREAM_H__
#define __FEATURE_STREAM_H__

#include "lib/common.h"
#include "base/SGObject.h"
#include "lib/io.h"
#include "lib/DataType.h"

namespace shogun
{
class CFeatureStream: public CSGObject
{
public:
	/** 
	 * Default constructor
	 */
	CFeatureStream() {}

	/** 
	 * Destructor
	 */
	virtual ~CFeatureStream() {}

	/** 
	 * Functions to read vectors.
	 *
	 * Set vector and length by reference.
	 * @param vector vector
	 * @param len length of vector
	 */
	virtual void get_bool_vector(bool*& vector, int32_t& len);
	virtual void get_byte_vector(uint8_t*& vector, int32_t& len);
	virtual void get_char_vector(char*& vector, int32_t& len);
	virtual void get_int_vector(int32_t*& vector, int32_t& len);
	virtual void get_real_vector(float64_t*& vector, int32_t& len);
	virtual void get_shortreal_vector(float32_t*& vector, int32_t& len);
	virtual void get_short_vector(int16_t*& vector, int32_t& len);
	virtual void get_word_vector(uint16_t*& vector, int32_t& len);
	virtual void get_int8_vector(int8_t*& vector, int32_t& len);
	virtual void get_uint_vector(uint32_t*& vector, int32_t& len);
	virtual void get_long_vector(int64_t*& vector, int32_t& len);
	virtual void get_ulong_vector(uint64_t*& vector, int32_t& len);
	virtual void get_longreal_vector(floatmax_t*& vector, int32_t& len);

	virtual void get_bool_vector(SGSparseVectorEntry<bool>*& vector, int32_t& len);
	virtual void get_byte_vector(SGSparseVectorEntry<uint8_t>*& vector, int32_t& len);
	virtual void get_char_vector(SGSparseVectorEntry<char>*& vector, int32_t& len);
	virtual void get_int_vector(SGSparseVectorEntry<int32_t>*& vector, int32_t& len);
	virtual void get_real_vector(SGSparseVectorEntry<float64_t>*& vector, int32_t& len);
	virtual void get_shortreal_vector(SGSparseVectorEntry<float32_t>*& vector, int32_t& len);
	virtual void get_short_vector(SGSparseVectorEntry<int16_t>*& vector, int32_t& len);
	virtual void get_word_vector(SGSparseVectorEntry<uint16_t>*& vector, int32_t& len);
	virtual void get_int8_vector(SGSparseVectorEntry<int8_t>*& vector, int32_t& len);
	virtual void get_uint_vector(SGSparseVectorEntry<uint32_t>*& vector, int32_t& len);
	virtual void get_long_vector(SGSparseVectorEntry<int64_t>*& vector, int32_t& len);
	virtual void get_ulong_vector(SGSparseVectorEntry<uint64_t>*& vector, int32_t& len);
	virtual void get_longreal_vector(SGSparseVectorEntry<floatmax_t>*& vector, int32_t& len);

	/** @name Label and Vector Access Functions
	 *
	 * Functions to access the label and vectors of examples
	 * one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector, its length, and the label by reference
	 */
	//@{
	virtual void get_bool_vector_and_label(bool*& vector, int32_t& len, float64_t& label);
	virtual void get_byte_vector_and_label(uint8_t*& vector, int32_t& len, float64_t& label);
	virtual void get_char_vector_and_label(char*& vector, int32_t& len, float64_t& label);
	virtual void get_int_vector_and_label(int32_t*& vector, int32_t& len, float64_t& label);
	virtual void get_real_vector_and_label(float64_t*& vector, int32_t& len, float64_t& label);
	virtual void get_shortreal_vector_and_label(float32_t*& vector, int32_t& len, float64_t& label);
	virtual void get_short_vector_and_label(int16_t*& vector, int32_t& len, float64_t& label);
	virtual void get_word_vector_and_label(uint16_t*& vector, int32_t& len, float64_t& label);
	virtual void get_int8_vector_and_label(int8_t*& vector, int32_t& len, float64_t& label);
	virtual void get_uint_vector_and_label(uint32_t*& vector, int32_t& len, float64_t& label);
	virtual void get_long_vector_and_label(int64_t*& vector, int32_t& len, float64_t& label);
	virtual void get_ulong_vector_and_label(uint64_t*& vector, int32_t& len, float64_t& label);
	virtual void get_longreal_vector_and_label(floatmax_t*& vector, int32_t& len, float64_t& label);

	virtual void get_bool_vector_and_label(SGSparseVectorEntry<bool>*& vector, int32_t& len, float64_t& label);
	virtual void get_byte_vector_and_label(SGSparseVectorEntry<uint8_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_char_vector_and_label(SGSparseVectorEntry<char>*& vector, int32_t& len, float64_t& label);
	virtual void get_int_vector_and_label(SGSparseVectorEntry<int32_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_real_vector_and_label(SGSparseVectorEntry<float64_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_shortreal_vector_and_label(SGSparseVectorEntry<float32_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_short_vector_and_label(SGSparseVectorEntry<int16_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_word_vector_and_label(SGSparseVectorEntry<uint16_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_int8_vector_and_label(SGSparseVectorEntry<int8_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_uint_vector_and_label(SGSparseVectorEntry<uint32_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_long_vector_and_label(SGSparseVectorEntry<int64_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_ulong_vector_and_label(SGSparseVectorEntry<uint64_t>*& vector, int32_t& len, float64_t& label);
	virtual void get_longreal_vector_and_label(SGSparseVectorEntry<floatmax_t>*& vector, int32_t& len, float64_t& label);

	//@}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "FeatureStream";

	}
};

#define GET_VECTOR_DUMMY(fname, sg_type)			\
void CFeatureStream::fname(sg_type*& vector, int32_t& len)	\
{								\
	SG_NOTIMPLEMENTED;					\
	return;							\
}								
	
GET_VECTOR_DUMMY(get_bool_vector, bool)
GET_VECTOR_DUMMY(get_byte_vector, uint8_t)
GET_VECTOR_DUMMY(get_char_vector, char)
GET_VECTOR_DUMMY(get_int_vector, int32_t)
GET_VECTOR_DUMMY(get_shortreal_vector, float32_t)
GET_VECTOR_DUMMY(get_real_vector, float64_t)
GET_VECTOR_DUMMY(get_short_vector, int16_t)
GET_VECTOR_DUMMY(get_word_vector, uint16_t)
GET_VECTOR_DUMMY(get_int8_vector, int8_t)
GET_VECTOR_DUMMY(get_uint_vector, uint32_t)
GET_VECTOR_DUMMY(get_long_vector, int64_t)
GET_VECTOR_DUMMY(get_ulong_vector, uint64_t)
GET_VECTOR_DUMMY(get_longreal_vector, floatmax_t)
#undef GET_VECTOR_DUMMY

#define GET_SPARSE_VECTOR_DUMMY(fname, sg_type)					\
void CFeatureStream::fname(SGSparseVectorEntry<sg_type>*& vector, int32_t& len)	\
{									\
	SG_NOTIMPLEMENTED;						\
	return;								\
}								
	
GET_SPARSE_VECTOR_DUMMY(get_bool_vector, bool)
GET_SPARSE_VECTOR_DUMMY(get_byte_vector, uint8_t)
GET_SPARSE_VECTOR_DUMMY(get_char_vector, char)
GET_SPARSE_VECTOR_DUMMY(get_int_vector, int32_t)
GET_SPARSE_VECTOR_DUMMY(get_shortreal_vector, float32_t)
GET_SPARSE_VECTOR_DUMMY(get_real_vector, float64_t)
GET_SPARSE_VECTOR_DUMMY(get_short_vector, int16_t)
GET_SPARSE_VECTOR_DUMMY(get_word_vector, uint16_t)
GET_SPARSE_VECTOR_DUMMY(get_int8_vector, int8_t)
GET_SPARSE_VECTOR_DUMMY(get_uint_vector, uint32_t)
GET_SPARSE_VECTOR_DUMMY(get_long_vector, int64_t)
GET_SPARSE_VECTOR_DUMMY(get_ulong_vector, uint64_t)
GET_SPARSE_VECTOR_DUMMY(get_longreal_vector, floatmax_t)
#undef GET_SPARSE_VECTOR_DUMMY

#define GET_VECTOR_AND_LABEL_DUMMY(fname, sg_type)			\
void CFeatureStream::fname(sg_type*& vector, int32_t& len, float64_t& label)	\
{									\
	SG_NOTIMPLEMENTED;						\
	return;								\
}								
	
GET_VECTOR_AND_LABEL_DUMMY(get_bool_vector_and_label, bool)
GET_VECTOR_AND_LABEL_DUMMY(get_byte_vector_and_label, uint8_t)
GET_VECTOR_AND_LABEL_DUMMY(get_char_vector_and_label, char)
GET_VECTOR_AND_LABEL_DUMMY(get_int_vector_and_label, int32_t)
GET_VECTOR_AND_LABEL_DUMMY(get_shortreal_vector_and_label, float32_t)
GET_VECTOR_AND_LABEL_DUMMY(get_real_vector_and_label, float64_t)
GET_VECTOR_AND_LABEL_DUMMY(get_short_vector_and_label, int16_t)
GET_VECTOR_AND_LABEL_DUMMY(get_word_vector_and_label, uint16_t)
GET_VECTOR_AND_LABEL_DUMMY(get_int8_vector_and_label, int8_t)
GET_VECTOR_AND_LABEL_DUMMY(get_uint_vector_and_label, uint32_t)
GET_VECTOR_AND_LABEL_DUMMY(get_long_vector_and_label, int64_t)
GET_VECTOR_AND_LABEL_DUMMY(get_ulong_vector_and_label, uint64_t)
GET_VECTOR_AND_LABEL_DUMMY(get_longreal_vector_and_label, floatmax_t)
#undef GET_VECTOR_AND_LABEL_DUMMY

#define GET_SPARSE_VECTOR_AND_LABEL_DUMMY(fname, sg_type)			\
void CFeatureStream::fname(SGSparseVectorEntry<sg_type>*& vector,	\
			   int32_t& len, float64_t& label)		\
{									\
	SG_NOTIMPLEMENTED;						\
	return;								\
}								
	
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_bool_vector_and_label, bool)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_byte_vector_and_label, uint8_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_char_vector_and_label, char)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_int_vector_and_label, int32_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_shortreal_vector_and_label, float32_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_real_vector_and_label, float64_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_short_vector_and_label, int16_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_word_vector_and_label, uint16_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_int8_vector_and_label, int8_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_uint_vector_and_label, uint32_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_long_vector_and_label, int64_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_ulong_vector_and_label, uint64_t)
GET_SPARSE_VECTOR_AND_LABEL_DUMMY(get_longreal_vector_and_label, floatmax_t)
#undef GET_SPARSE_VECTOR_AND_LABEL_DUMMY

	
}
#endif //__FEATURE_STREAM_H__
