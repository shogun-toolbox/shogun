/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Yuyu Zhang,
 *          Evangelos Anagnostopoulos, Sergey Lisitsyn
 */
#ifndef __STREAMING_FILEFROMSTRING_H__
#define __STREAMING_FILEFROMSTRING_H__

#include <shogun/lib/config.h>

#include <shogun/io/streaming/StreamingFileFromFeatures.h>
#include <shogun/features/StringFeatures.h>

namespace shogun
{
/** @brief Class StreamingFileFromStringFeatures is derived from
 * CStreamingFile and provides an input source for the online
 * framework from a CStringFeatures object.
 */
template <class T> class StreamingFileFromStringFeatures: public StreamingFileFromFeatures
{
public:
	/**
	 * Default constructor
	 */
	StreamingFileFromStringFeatures();

	/**
	 * Constructor taking a StringFeatures object as arg
	 *
	 * @param feat StringFeatures object
	 */
	StreamingFileFromStringFeatures(std::shared_ptr<StringFeatures<T>> feat);

	/**
	 * Constructor taking a StringFeatures object as arg
	 *
	 * @param feat StringFeatures object
	 * @param lab Labels as float64_t*
	 */
	StreamingFileFromStringFeatures(std::shared_ptr<StringFeatures<T>> feat, float64_t* lab);

	/**
	 * Destructor
	 */
	virtual ~StreamingFileFromStringFeatures();

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
	void init(std::shared_ptr<StringFeatures<T>> feat=NULL);

protected:

	/// StringFeatures object
	std::shared_ptr<StringFeatures<T>> features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template <class T>
StreamingFileFromStringFeatures<T>::StreamingFileFromStringFeatures()
	: StreamingFileFromFeatures()
{
	init();
}

template <class T>
StreamingFileFromStringFeatures<T>::StreamingFileFromStringFeatures(std::shared_ptr<StringFeatures<T>> feat)
	: StreamingFileFromFeatures(feat)
{
	init(feat);
}

template <class T>
StreamingFileFromStringFeatures<T>::StreamingFileFromStringFeatures(std::shared_ptr<StringFeatures<T>> feat, float64_t* lab)
	: StreamingFileFromFeatures(feat,lab)
{
	init(feat);
}

template <class T>
StreamingFileFromStringFeatures<T>::~StreamingFileFromStringFeatures()
{
}

template <class T>
void StreamingFileFromStringFeatures<T>::init(std::shared_ptr<StringFeatures<T>> feat)
{
	vector_num=0;
	features = feat;

	set_generic<T>();
}

/* Functions to return the vector from the StringFeatures object */
template <class T>
void StreamingFileFromStringFeatures<T>::get_string(T*& vector, int32_t& num_feat)
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
void StreamingFileFromStringFeatures<T>::get_string_and_label
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
