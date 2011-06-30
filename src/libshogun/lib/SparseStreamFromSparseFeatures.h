/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILEFROMSPARSE_H__
#define __STREAMING_FILEFROMSPARSE_H__

#include "lib/SparseStream.h"
#include "features/SparseFeatures.h"

namespace shogun
{
class CSparseStreamFromSparseFeatures: public CSparseStream
{
public:
	/** 
	 * Default constructor
	 */
	CSparseStreamFromSparseFeatures();

	/** 
	 * Constructor taking a SparseFeatures object as arg
	 * 
	 * @param feat SparseFeatures object
	 */
	CSparseStreamFromSparseFeatures(CFeatures* feat);

	/** 
	 * Constructor taking a SparseFeatures object, and labels as args
	 * 
	 * @param feat SparseFeatures object
	 * @param labels labels as float64_t*
	 */
	CSparseStreamFromSparseFeatures(CFeatures* feat, float64_t* labels);

	/** 
	 * Destructor
	 */
	virtual ~CSparseStreamFromSparseFeatures();

	/** 
	 * Functions to read vectors from the SparseFeatures object
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
	//@}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "SparseStreamFromSparseFeatures";

	}

private:
	/** 
	 * Initialize members to defaults
	 */
	void init();
	
protected:
	/// The SparseFeatures object
	CFeatures* features;

	// Labels (if any)
	float64_t* labels;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};
}
#endif //__STREAMING_FILEFROMSPARSE_H__
