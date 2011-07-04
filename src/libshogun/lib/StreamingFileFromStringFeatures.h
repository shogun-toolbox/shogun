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
template <class T> class CStreamingFileFromStringFeatures: public CStreamingFileFromFeatures
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
	CStreamingFileFromStringFeatures(CStringFeatures<T>* feat);

	/**
	 * Constructor taking a StringFeatures object as arg
	 *
	 * @param feat StringFeatures object
	 * @param lab Labels as float64_t*
	 */
	CStreamingFileFromStringFeatures(CStringFeatures<T>* feat, float64_t* lab);

	/**
	 * Destructor
	 */
	virtual ~CStreamingFileFromStringFeatures();

	/**
	 * This function will be called for reading strings from the
	 * corresponding StringFeatures object.
	 * It is specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 */
	virtual void get_string(T* &vec, int32_t &len);

	/**
	 * This function will be called for reading strings and labels
	 * from the corresponding StringFeatures object.  It is
	 * specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 * @param label label
	 */
	virtual void get_string_and_label(T* &vec, int32_t &len, float64_t &label);

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
		return "StreamingFileFromStringFeatures";

	}

private:
	/**
	 * Initialize members to defaults
	 */
	void init();

protected:

	/// StringFeatures object
	CStringFeatures<T>* features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template <class T>
CStreamingFileFromStringFeatures<T>::CStreamingFileFromStringFeatures()
	: CStreamingFileFromFeatures()
{
	init();
}

template <class T>
CStreamingFileFromStringFeatures<T>::CStreamingFileFromStringFeatures(CStringFeatures<T>* feat)
	: CStreamingFileFromFeatures(feat)
{
	init();
}

template <class T>
CStreamingFileFromStringFeatures<T>::CStreamingFileFromStringFeatures(CStringFeatures<T>* feat, float64_t* lab)
	: CStreamingFileFromFeatures(feat,lab)
{
	init();
}

template <class T>
CStreamingFileFromStringFeatures<T>::~CStreamingFileFromStringFeatures()
{
}

template <class T>
void CStreamingFileFromStringFeatures<T>::init()
{
	vector_num=0;
}

/* Functions to return the vector from the StringFeatures object */
#define GET_STRING(sg_type)					\
	template <>								\
	void CStreamingFileFromStringFeatures<sg_type>::get_string(sg_type*& vector, int32_t& num_feat) \
	{								\
		if (vector_num >= features->get_num_vectors())		\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector=				\
			features->get_feature_vector(vector_num);	\
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;;				\
		vector_num++;						\
									\
	}								\

GET_STRING(bool)
GET_STRING(uint8_t)
GET_STRING(char)
GET_STRING(int32_t)
GET_STRING(float32_t)
GET_STRING(float64_t)
GET_STRING(int16_t)
GET_STRING(uint16_t)
GET_STRING(int8_t)
GET_STRING(uint32_t)
GET_STRING(int64_t)
GET_STRING(uint64_t)
GET_STRING(floatmax_t)
#undef GET_STRING

/* Functions to return the vector from the StringFeatures object with label */
#define GET_STRING_AND_LABEL(sg_type)					\
	template <>							\
	void CStreamingFileFromStringFeatures<sg_type>::get_string_and_label\
	(sg_type*& vector, int32_t& num_feat, float64_t& label)		\
	{								\
		if (vector_num >= features->get_num_vectors())		\
		{							\
			vector=NULL;					\
			num_feat=-1;					\
			return;						\
		}							\
									\
		SGVector<sg_type> sg_vector				\
			=features->get_feature_vector(vector_num);		\
									\
		vector = sg_vector.vector;				\
		num_feat = sg_vector.vlen;				\
		label = labels[vector_num];				\
									\
		vector_num++;						\
	}								\

GET_STRING_AND_LABEL(bool)
GET_STRING_AND_LABEL(uint8_t)
GET_STRING_AND_LABEL(char)
GET_STRING_AND_LABEL(int32_t)
GET_STRING_AND_LABEL(float32_t)
GET_STRING_AND_LABEL(float64_t)
GET_STRING_AND_LABEL(int16_t)
GET_STRING_AND_LABEL(uint16_t)
GET_STRING_AND_LABEL(int8_t)
GET_STRING_AND_LABEL(uint32_t)
GET_STRING_AND_LABEL(int64_t)
GET_STRING_AND_LABEL(uint64_t)
GET_STRING_AND_LABEL(floatmax_t)
#undef GET_STRING_AND_LABEL

}
#endif //__STREAMING_FILEFROMSTRING_H__
