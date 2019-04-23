/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Yuyu Zhang,
 *          Evangelos Anagnostopoulos, Sergey Lisitsyn
 */
#ifndef __STREAMING_FILEFROMDENSE_H__
#define __STREAMING_FILEFROMDENSE_H__

#include <shogun/lib/config.h>

#include <shogun/io/streaming/StreamingFileFromFeatures.h>
#include <shogun/features/DenseFeatures.h>

namespace shogun
{
/** @brief Class StreamingFileFromDenseFeatures is a derived
 * class of CStreamingFile which creates an input source
 * for the online framework from a DenseFeatures object.
 *
 * This kind of input is seekable, and hence can be used for
 * making multiple passes over data.
 *
 * It is useful for testing/comparison purposes.
 */
template<class T> class StreamingFileFromDenseFeatures:
	public StreamingFileFromFeatures
{
public:
	/**
	 * Default constructor
	 */
	StreamingFileFromDenseFeatures();

	/**
	 * Constructor taking a DenseFeatures object as arg
	 *
	 * @param feat DenseFeatures object
	 * @param lab Labels as float64_t*, optional
	 */
	StreamingFileFromDenseFeatures(std::shared_ptr<DenseFeatures<T>> feat,
			float64_t* lab=NULL);

	/**
	 * Destructor
	 */
	virtual ~StreamingFileFromDenseFeatures();

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
	std::shared_ptr<DenseFeatures<T>> features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template<class T>
StreamingFileFromDenseFeatures<T>::StreamingFileFromDenseFeatures() :
		StreamingFileFromFeatures()
{
	init();
}

template<class T>
StreamingFileFromDenseFeatures<T>::StreamingFileFromDenseFeatures(
		std::shared_ptr<DenseFeatures<T>> feat, float64_t* lab) :
		StreamingFileFromDenseFeatures()
{
	require(feat,"{}::StreamingFileFromDenseFeatures() features required!",
			get_name());
	features=feat;
	labels=lab;
}

template<class T>
StreamingFileFromDenseFeatures<T>::~StreamingFileFromDenseFeatures()
{
}

template<class T>
void StreamingFileFromDenseFeatures<T>::init()
{
	vector_num=0;
	features=NULL;

	set_generic<T>();
}

/* Functions to return the vector from the DenseFeatures object
 * If the class is of type T, specialize this function to work for
 * vectors of that type. */
template<class T>
void StreamingFileFromDenseFeatures<T>::get_vector(T*& vector,
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
void StreamingFileFromDenseFeatures<T>::get_vector_and_label(T*& vector,
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
