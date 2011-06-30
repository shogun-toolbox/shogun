/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_SPARSEASCII_H__
#define __STREAMING_SPARSEASCII_H__

#include "lib/SparseStream.h"
#include "lib/AsciiFile.h"

namespace shogun
{
class CSparseAsciiStream: public CSparseStream
{
public:

	/** 
	 * Default constructor.
	 */
	CSparseAsciiStream();
	
	/** 
	 * Constructor taking an AsciiFile object as parameter
	 * 
	 * @param f CAsciiFile object from which to read
	 */
	CSparseAsciiStream(CAsciiFile* f);

	/** 
	 * Destructor
	 */
	virtual ~CSparseAsciiStream();

	/** 
	 * Functions to read a sparse vector from an ASCII file.
	 *
	 * Set vector and length by reference.
	 * @param vector vector
	 * @param len length of vector
	 */
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
	
	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "SparseAsciiStream";

	}

private:
	/** 
	 * Utility function to convert a string to bool.
	 * 
	 * @param str char* pointer
	 * 
	 * @return boolean conversion of the string
	 */
	inline bool str_to_bool(char *str);

protected:
	/// The CAsciiFile object from which to read
	CAsciiFile* ascii_file;
};

}
#endif //__STREAMING_SPARSEASCII_H__
