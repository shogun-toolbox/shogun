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

#include "lib/StreamingFileFromFeatures.h"
#include "features/SparseFeatures.h"

namespace shogun
{
template <class T> class CStreamingFileFromSparseFeatures: public CStreamingFileFromFeatures
{
public:
	/**
	 * Default constructor
	 */
	CStreamingFileFromSparseFeatures();

	/**
	 * Constructor taking a SparseFeatures object as arg
	 *
	 * @param feat SparseFeatures object
	 */
	CStreamingFileFromSparseFeatures(CSparseFeatures<T>* feat);

	/**
	 * Constructor taking a SparseFeatures object as arg
	 *
	 * @param feat SparseFeatures object
	 * @param lab Labels as float64_t*
	 */
	CStreamingFileFromSparseFeatures(CSparseFeatures<T>* feat, float64_t* lab);

	/**
	 * Destructor
	 */
	virtual ~CStreamingFileFromSparseFeatures();

	/**
	 * This function will be called for reading vectors from the
	 * corresponding SparseFeatures object.
	 * It is specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 */
	virtual void get_sparse_vector(SGSparseVectorEntry<T>* &vec, int32_t &len);

	/**
	 * This function will be called for reading vectors and labels
	 * from the corresponding SparseFeatures object.  It is
	 * specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 * @param label label
	 */
	virtual void get_sparse_vector_and_label(SGSparseVectorEntry<T>* &vec, int32_t &len, float64_t &label);

	/**
	 * Reset the stream so the next example returned is the first
	 * example in the SparseFeatures object.
	 *
	 */
	void reset_stream()
	{
		vector_num = 0;
	}

	/** @return object name */
	inline virtual const char* get_name() const
	{
		return "StreamingFileFromSparseFeatures";

	}

private:
	/**
	 * Initialize members to defaults
	 */
	void init();

protected:
	/// SparseFeatures object
	CSparseFeatures<T>* features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template <class T>
CStreamingFileFromSparseFeatures<T>::CStreamingFileFromSparseFeatures()
	: CStreamingFileFromFeatures()
{
	init();
}

template <class T>
CStreamingFileFromSparseFeatures<T>::CStreamingFileFromSparseFeatures(CSparseFeatures<T>* feat)
	: CStreamingFileFromFeatures(feat)
{
	init();
}

template <class T>
CStreamingFileFromSparseFeatures<T>::CStreamingFileFromSparseFeatures(CSparseFeatures<T>* feat, float64_t* lab)
	: CStreamingFileFromFeatures(feat,lab)
{
	init();
}

template <class T>
CStreamingFileFromSparseFeatures<T>::~CStreamingFileFromSparseFeatures()
{
}

template <class T>
void CStreamingFileFromSparseFeatures<T>::init()
{
	vector_num=0;
}

/* Functions to return the vector from the SparseFeatures object */
template <class T>
void CStreamingFileFromSparseFeatures<T>::get_sparse_vector
(SGSparseVectorEntry<T>*& vector, int32_t& len)
{
	if (vector_num >= features->get_num_vectors())
	{
		vector=NULL;
		len=-1;
		return;
	}

	bool vfree;
	vector=features->get_sparse_feature_vector
		(vector_num, len, vfree);

	vector_num++;
}

/* Functions to return the vector from the SparseFeatures object */
template <class T>
void CStreamingFileFromSparseFeatures<T>::get_sparse_vector_and_label
(SGSparseVectorEntry<T>*& vector, int32_t& len, float64_t& label)
{
	if (vector_num >= features->get_num_vectors())
	{
		vector=NULL;
		len=-1;
		return;
	}

	bool vfree;
	vector=features->get_sparse_feature_vector
		(vector_num, len, vfree);
	label=labels[vector_num];

	vector_num++;
}

}
#endif //__STREAMING_FILEFROMSPARSE_H__
