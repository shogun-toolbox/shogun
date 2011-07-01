/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILEFROMSTRING_H__
#define __STREAMING_FILEFROMSTRING_H__

#include "lib/StreamingFileFromFeatures.h"
#include "features/StringFeatures.h"

namespace shogun
{
class CStreamingFileFromStringFeatures: public CStreamingFileFromFeatures
{
public:
	/** 
	 * Default constructor
	 */
	CStreamingFileFromStringFeatures();

	/** 
	 * Constructor taking a StringFeatures object as arg
	 * 
	 * @param feat StringFeatures object
	 */
	CStreamingFileFromStringFeatures(CFeatures* feat);
	
	/** 
	 * Constructor taking a StringFeatures object as arg
	 * 
	 * @param feat StringFeatures object
	 * @param lab Labels as float64_t*
	 */
	CStreamingFileFromStringFeatures(CFeatures* feat, float64_t* lab);

	/** 
	 * Destructor
	 */
	virtual ~CStreamingFileFromStringFeatures();

	/** 
	 * Functions to read vectors from the StringFeatures object
	 *
	 * Set vector and length by reference.
	 * @param vector vector
	 * @param len length of vector
	 */
	virtual void get_bool_string(bool*& vector, int32_t& len);
	virtual void get_byte_string(uint8_t*& vector, int32_t& len);
	virtual void get_char_string(char*& vector, int32_t& len);
	virtual void get_int_string(int32_t*& vector, int32_t& len);
	virtual void get_real_string(float64_t*& vector, int32_t& len);
	virtual void get_shortreal_string(float32_t*& vector, int32_t& len);
	virtual void get_short_string(int16_t*& vector, int32_t& len);
	virtual void get_word_string(uint16_t*& vector, int32_t& len);
	virtual void get_int8_string(int8_t*& vector, int32_t& len);
	virtual void get_uint_string(uint32_t*& vector, int32_t& len);
	virtual void get_long_string(int64_t*& vector, int32_t& len);
	virtual void get_ulong_string(uint64_t*& vector, int32_t& len);
	virtual void get_longreal_string(floatmax_t*& vector, int32_t& len);

	/** @name Label and Vector Access Functions
	 *
	 * Functions to access the label and vectors of examples
	 * one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector, its length, and the label by reference
	 */
	//@{
	virtual void get_bool_string_and_label(bool*& vector, int32_t& len, float64_t& label);
	virtual void get_byte_string_and_label(uint8_t*& vector, int32_t& len, float64_t& label);
	virtual void get_char_string_and_label(char*& vector, int32_t& len, float64_t& label);
	virtual void get_int_string_and_label(int32_t*& vector, int32_t& len, float64_t& label);
	virtual void get_real_string_and_label(float64_t*& vector, int32_t& len, float64_t& label);
	virtual void get_shortreal_string_and_label(float32_t*& vector, int32_t& len, float64_t& label);
	virtual void get_short_string_and_label(int16_t*& vector, int32_t& len, float64_t& label);
	virtual void get_word_string_and_label(uint16_t*& vector, int32_t& len, float64_t& label);
	virtual void get_int8_string_and_label(int8_t*& vector, int32_t& len, float64_t& label);
	virtual void get_uint_string_and_label(uint32_t*& vector, int32_t& len, float64_t& label);
	virtual void get_long_string_and_label(int64_t*& vector, int32_t& len, float64_t& label);
	virtual void get_ulong_string_and_label(uint64_t*& vector, int32_t& len, float64_t& label);
	virtual void get_longreal_string_and_label(floatmax_t*& vector, int32_t& len, float64_t& label);
	//@}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "StreamingFileFromStringFeatures";

	}

private:
	/** 
	 * Initialize members to defaults
	 */
	void init();
	
protected:

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};
}
#endif //__STREAMING_FILEFROMSTRING_H__
