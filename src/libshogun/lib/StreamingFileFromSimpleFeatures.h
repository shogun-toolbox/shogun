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
template <class T> class CStreamingFileFromSimpleFeatures: public CStreamingFileFromFeatures
{
public:
	/**
	 * Default constructor
	 */
	CStreamingFileFromSimpleFeatures();

	/**
	 * Constructor taking a SimpleFeatures object as arg
	 *
	 * @param feat SimpleFeatures object
	 */
	CStreamingFileFromSimpleFeatures(CSimpleFeatures<T>* feat);

	/**
	 * Constructor taking a SimpleFeatures object as arg
	 *
	 * @param feat SimpleFeatures object
	 * @param lab Labels as float64_t*
	 */
	CStreamingFileFromSimpleFeatures(CSimpleFeatures<T>* feat, float64_t* lab);

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
	virtual void get_vector(bool*& vector, int32_t& len);
	virtual void get_vector(uint8_t*& vector, int32_t& len);
	virtual void get_vector(char*& vector, int32_t& len);
	virtual void get_vector(int32_t*& vector, int32_t& len);
	virtual void get_vector(float64_t*& vector, int32_t& len);
	virtual void get_vector(float32_t*& vector, int32_t& len);
	virtual void get_vector(int16_t*& vector, int32_t& len);
	virtual void get_vector(uint16_t*& vector, int32_t& len);
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

	/**
	 * Reset the stream so the next example returned is the first
	 * example in the SimpleFeatures object.
	 *
	 */
	void reset_stream()
	{
		vector_num = 0;
	}

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

	/// SimpleFeatures object
	CSimpleFeatures<T>* features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template <class T>
CStreamingFileFromSimpleFeatures<T>::CStreamingFileFromSimpleFeatures()
	: CStreamingFileFromFeatures()
{
	init();
}

template <class T>
CStreamingFileFromSimpleFeatures<T>::CStreamingFileFromSimpleFeatures(CSimpleFeatures<T>* feat)
	: CStreamingFileFromFeatures()
{
	ASSERT(feat);
	features=feat;

	init();
}

template <class T>
CStreamingFileFromSimpleFeatures<T>::CStreamingFileFromSimpleFeatures(CSimpleFeatures<T>* feat, float64_t* lab)
	: CStreamingFileFromFeatures()
{
	ASSERT(feat);
	ASSERT(lab);
	features=feat;
	labels=lab;

	init();
}

template <class T>
CStreamingFileFromSimpleFeatures<T>::~CStreamingFileFromSimpleFeatures()
{
}

template <class T>
void CStreamingFileFromSimpleFeatures<T>::init()
{
	vector_num=0;
}

/* Functions to return the vector from the SimpleFeatures object */
#define GET_VECTOR(fname, sg_type)					\
	template <class T>						\
	void CStreamingFileFromSimpleFeatures<T>::fname(sg_type*& vector, int32_t& num_feat) \
	{								\
		CSimpleFeatures<sg_type>* simple_features=		\
			(CSimpleFeatures<sg_type>*) features;		\
									\
		if (vector_num >= simple_features->get_num_vectors())	\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector=				\
			simple_features->get_feature_vector(vector_num); \
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;;				\
		vector_num++;						\
									\
	}								\

GET_VECTOR(get_bool_vector, bool)
GET_VECTOR(get_byte_vector, uint8_t)
GET_VECTOR(get_char_vector, char)
GET_VECTOR(get_int_vector, int32_t)
GET_VECTOR(get_shortreal_vector, float32_t)
GET_VECTOR(get_real_vector, float64_t)
GET_VECTOR(get_short_vector, int16_t)
GET_VECTOR(get_word_vector, uint16_t)
GET_VECTOR(get_int8_vector, int8_t)
GET_VECTOR(get_uint_vector, uint32_t)
GET_VECTOR(get_long_vector, int64_t)
GET_VECTOR(get_ulong_vector, uint64_t)
GET_VECTOR(get_longreal_vector, floatmax_t)
#undef GET_VECTOR

/* Functions to return the vector from the SimpleFeatures object with label */
#define GET_VECTOR_AND_LABEL(fname, sg_type)				\
	template <class T>						\
	void CStreamingFileFromSimpleFeatures<T>::fname			\
	(sg_type*& vector, int32_t& num_feat, float64_t& label)		\
	{								\
		CSimpleFeatures<sg_type>* feat				\
			=(CSimpleFeatures<sg_type>*) features;		\
									\
		if (vector_num >= feat->get_num_vectors())		\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector				\
			=feat->get_feature_vector(vector_num);		\
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;				\
		label = labels[vector_num];				\
									\
		vector_num++;						\
	}								\

GET_VECTOR_AND_LABEL(get_bool_vector_and_label, bool)
GET_VECTOR_AND_LABEL(get_byte_vector_and_label, uint8_t)
GET_VECTOR_AND_LABEL(get_char_vector_and_label, char)
GET_VECTOR_AND_LABEL(get_int_vector_and_label, int32_t)
GET_VECTOR_AND_LABEL(get_shortreal_vector_and_label, float32_t)
GET_VECTOR_AND_LABEL(get_real_vector_and_label, float64_t)
GET_VECTOR_AND_LABEL(get_short_vector_and_label, int16_t)
GET_VECTOR_AND_LABEL(get_word_vector_and_label, uint16_t)
GET_VECTOR_AND_LABEL(get_int8_vector_and_label, int8_t)
GET_VECTOR_AND_LABEL(get_uint_vector_and_label, uint32_t)
GET_VECTOR_AND_LABEL(get_long_vector_and_label, int64_t)
GET_VECTOR_AND_LABEL(get_ulong_vector_and_label, uint64_t)
GET_VECTOR_AND_LABEL(get_longreal_vector_and_label, floatmax_t)
#undef GET_VECTOR_AND_LABEL

}
#endif //__STREAMING_FILEFROMSIMPLE_H__
