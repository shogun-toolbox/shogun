/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILEFROMSIMPLE_H__
#define __STREAMING_FILEFROMSIMPLE_H__

#include "lib/StreamingFileFromFeatures.h"
#include "features/SimpleFeatures.h"

namespace shogun
{
class CStreamingFileFromSimpleFeatures: public CStreamingFileFromFeatures
{
public:
	/** 
	 * Default constructor
	 */
	CStreamingFileFromSimpleFeatures(void);

	/** 
	 * Constructor taking a SimpleFeatures object as arg
	 * 
	 * @param feat SimpleFeatures object
	 */
	CStreamingFileFromSimpleFeatures(CFeatures* feat);

	/** 
	 * Destructor
	 */
	virtual ~CStreamingFileFromSimpleFeatures();

	/** 
	 * Functions to read vectors from the SimpleFeatures object
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

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "StreamingFileFromSimpleFeatures";

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
#endif //__STREAMING_FILEFROMSIMPLE_H__
