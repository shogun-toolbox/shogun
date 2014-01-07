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

#include <io/streaming/StreamingFileFromFeatures.h>
#include <features/StringFeatures.h>

namespace shogun
{
/** @brief Class CStreamingFileFromStringFeatures is derived from
 * CStreamingFile and provides an input source for the online
 * framework from a CStringFeatures object.
 */
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
	 * example in the DenseFeatures object.
	 *
	 */
	void reset_stream()
	{
		vector_num = 0;
	}

	/** @return object name */
	virtual const char* get_name() const
	{
		return "StreamingFileFromStringFeatures";

	}

private:
	/**
	 * Initialize members to defaults
	 */
	void init(CStringFeatures<T>* feat=NULL);

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
	init(feat);
}

template <class T>
CStreamingFileFromStringFeatures<T>::CStreamingFileFromStringFeatures(CStringFeatures<T>* feat, float64_t* lab)
	: CStreamingFileFromFeatures(feat,lab)
{
	init(feat);
}

template <class T>
CStreamingFileFromStringFeatures<T>::~CStreamingFileFromStringFeatures()
{
}

template <class T>
void CStreamingFileFromStringFeatures<T>::init(CStringFeatures<T>* feat)
{
	vector_num=0;
	features = feat;

	set_generic<T>();
}

/* Functions to return the vector from the StringFeatures object */
template <class T>
void CStreamingFileFromStringFeatures<T>::get_string(T*& vector, int32_t& num_feat)
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
	num_feat = sg_vector.vlen;
	sg_vector.vector = NULL;
	sg_vector.vlen = -1;
	vector_num++;
}

/* Functions to return the vector from the StringFeatures object with label */
template <class T>
void CStreamingFileFromStringFeatures<T>::get_string_and_label
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
#endif //__STREAMING_FILEFROMSTRING_H__
