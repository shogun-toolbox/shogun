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

#include <shogun/lib/StreamingFileFromFeatures.h>
#include <shogun/features/SimpleFeatures.h>

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
	 * This function will be called for reading vectors from the
	 * corresponding SimpleFeatures object.
	 * It is specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 */
	virtual void get_vector(T* &vec, int32_t &len);

	/**
	 * This function will be called for reading vectors and labels
	 * from the corresponding SimpleFeatures object.  It is
	 * specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 * @param label label
	 */
	virtual void get_vector_and_label(T* &vec, int32_t &len, float64_t &label);

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

/* Functions to return the vector from the SimpleFeatures object
 * If the class is of type T, specialize this function to work for
 * vectors of that type. */
template <class T>
void CStreamingFileFromSimpleFeatures<T>::get_vector(T*& vector, int32_t& num_feat)
{
	if (vector_num >= features->get_num_vectors())
	{
		vector=NULL;
		num_feat=-1;
		return;
	}

	SGVector<T> sg_vector=
		features->get_feature_vector(vector_num);

	vector = sg_vector.vector;
	num_feat = sg_vector.vlen;;
	vector_num++;

}

/* Functions to return the vector from the SimpleFeatures object with label */
template <class T>
void CStreamingFileFromSimpleFeatures<T>::get_vector_and_label
(T*& vector, int32_t& num_feat, float64_t& label)
{
	if (vector_num >= features->get_num_vectors())
	{
		vector=NULL;
		num_feat=-1;
		return;
	}

	SGVector<T> sg_vector
		=features->get_feature_vector(vector_num);

	vector = sg_vector.vector;
	num_feat = sg_vector.vlen;
	label = labels[vector_num];

	vector_num++;
}

}
#endif //__STREAMING_FILEFROMSIMPLE_H__
