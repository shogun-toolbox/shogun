/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_SIMPLEASCII_H__
#define __STREAMING_SIMPLEASCII_H__

#include "lib/SimpleStream.h"
#include "lib/AsciiFile.h"

namespace shogun
{
class CSimpleAsciiStream: public CSimpleStream
{
public:
	/** 
	 * Default constructor
	 */
	CSimpleAsciiStream();

	/** 
	 * Constructor taking AsciiFile object as parameter
	 * 
	 * @param f CAsciiFile object from which to read
	 */
	CSimpleAsciiStream(CAsciiFile* f);

	/** 
	 * Destructor
	 */
	virtual ~CSimpleAsciiStream();

	/** 
	 * Functions to read a dense vector from an ASCII file.
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
	//@}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "SimpleAsciiStream";

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
#endif //__STREAMING_SIMPLEASCII_H__
