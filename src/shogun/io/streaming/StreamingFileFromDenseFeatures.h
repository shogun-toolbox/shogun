/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILEFROMDENSE_H__
#define __STREAMING_FILEFROMDENSE_H__

#include <io/streaming/StreamingFileFromFeatures.h>
#include <features/DenseFeatures.h>

namespace shogun
{
/** @brief Class CStreamingFileFromDenseFeatures is a derived
 * class of CStreamingFile which creates an input source
 * for the online framework from a CDenseFeatures object.
 *
 * This kind of input is seekable, and hence can be used for
 * making multiple passes over data.
 *
 * It is useful for testing/comparison purposes.
 */
template<class T> class CStreamingFileFromDenseFeatures:
	public CStreamingFileFromFeatures
{
public:
	/**
	 * Default constructor
	 */
	CStreamingFileFromDenseFeatures();

	/**
	 * Constructor taking a DenseFeatures object as arg
	 *
	 * @param feat DenseFeatures object
	 * @param lab Labels as float64_t*, optional
	 */
	CStreamingFileFromDenseFeatures(CDenseFeatures<T>* feat,
			float64_t* lab=NULL);

	/**
	 * Destructor
	 */
	virtual ~CStreamingFileFromDenseFeatures();

	/**
	 * This function will be called for reading vectors from the
	 * corresponding DenseFeatures object.
	 * It is specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 */
	virtual void get_vector(T* &vec, int32_t &len);

	/**
	 * This function will be called for reading vectors and labels
	 * from the corresponding DenseFeatures object.  It is
	 * specialized depending on class type T.
	 *
	 * @param vec vector
	 * @param len length of vector
	 * @param label label
	 */
	virtual void get_vector_and_label(T* &vec, int32_t &len, float64_t &label);

	/**
	 * Reset the stream so the next example returned is the first
	 * example in the DenseFeatures object.
	 *
	 */
	void reset_stream()
	{
		vector_num=0;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "StreamingFileFromDenseFeatures";

	}

private:
	/**
	 * Initialize members to defaults
	 */
	void init();

protected:

	/// DenseFeatures object
	CDenseFeatures<T>* features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template<class T>
CStreamingFileFromDenseFeatures<T>::CStreamingFileFromDenseFeatures() :
		CStreamingFileFromFeatures()
{
	init();
}

template<class T>
CStreamingFileFromDenseFeatures<T>::CStreamingFileFromDenseFeatures(
		CDenseFeatures<T>* feat, float64_t* lab) :
		CStreamingFileFromFeatures()
{
	init();

	REQUIRE(feat,"%s::CStreamingFileFromDenseFeatures() features required!\n",
			get_name());
	features=feat;
	SG_REF(feat);

	labels=lab;

}

template<class T>
CStreamingFileFromDenseFeatures<T>::~CStreamingFileFromDenseFeatures()
{
	SG_UNREF(features);
}

template<class T>
void CStreamingFileFromDenseFeatures<T>::init()
{
	vector_num=0;
	features=NULL;

	set_generic<T>();
}

/* Functions to return the vector from the DenseFeatures object
 * If the class is of type T, specialize this function to work for
 * vectors of that type. */
template<class T>
void CStreamingFileFromDenseFeatures<T>::get_vector(T*& vector,
		int32_t& num_feat)
{
	if (vector_num>=features->get_num_vectors())
	{
		vector=NULL;
		num_feat=-1;
		return;
	}

	SGVector<T> sg_vector=features->get_feature_vector(vector_num);

	vector=sg_vector.vector;
	num_feat=sg_vector.vlen;
	vector_num++;

}

/* Functions to return the vector from the DenseFeatures object with label */
template<class T>
void CStreamingFileFromDenseFeatures<T>::get_vector_and_label(T*& vector,
		int32_t& num_feat, float64_t& label)
{
	if (vector_num>=features->get_num_vectors())
	{
		vector=NULL;
		num_feat=-1;
		return;
	}

	SGVector<T> sg_vector=features->get_feature_vector(vector_num);

	vector=sg_vector.vector;
	num_feat=sg_vector.vlen;
	label=labels[vector_num];

	vector_num++;
}


}
#endif //__STREAMING_FILEFROMDENSE_H__
