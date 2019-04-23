/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Yuyu Zhang,
 *          Evangelos Anagnostopoulos, Sergey Lisitsyn
 */
#ifndef __STREAMING_FILEFROMSPARSE_H__
#define __STREAMING_FILEFROMSPARSE_H__

#include <shogun/lib/config.h>

#include <shogun/io/streaming/StreamingFileFromFeatures.h>
#include <shogun/features/SparseFeatures.h>

namespace shogun
{
/** @brief Class StreamingFileFromSparseFeatures is derived from CStreamingFile
 * and provides an input source for the online framework. It uses an existing
 * SparseFeatures object to generate online examples.
 */
template <class T> class StreamingFileFromSparseFeatures: public StreamingFileFromFeatures
{
public:
	/**
	 * Default constructor
	 */
	StreamingFileFromSparseFeatures();

	/**
	 * Constructor taking a SparseFeatures object as arg
	 *
	 * @param feat SparseFeatures object
	 */
	StreamingFileFromSparseFeatures(std::shared_ptr<SparseFeatures<T>> feat);

	/**
	 * Constructor taking a SparseFeatures object as arg
	 *
	 * @param feat SparseFeatures object
	 * @param lab Labels as float64_t*
	 */
	StreamingFileFromSparseFeatures(std::shared_ptr<SparseFeatures<T>> feat, float64_t* lab);

	/**
	 * Destructor
	 */
	virtual ~StreamingFileFromSparseFeatures();

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
	virtual const char* get_name() const
	{
		return "StreamingFileFromSparseFeatures";

	}

private:
	/**
	 * Initialize members to defaults
	 */
	void init(std::shared_ptr<SparseFeatures<T>> feat);

protected:
	/// SparseFeatures object
	std::shared_ptr<SparseFeatures<T>> features;

	/// Index of vector to be returned from the feature matrix
	int32_t vector_num;

};

template <class T>
StreamingFileFromSparseFeatures<T>::StreamingFileFromSparseFeatures()
	: StreamingFileFromFeatures()
{
	init(NULL);
}

template <class T>
StreamingFileFromSparseFeatures<T>::StreamingFileFromSparseFeatures(std::shared_ptr<SparseFeatures<T>> feat)
	: StreamingFileFromFeatures(feat)
{
	init(feat);
}

template <class T>
StreamingFileFromSparseFeatures<T>::StreamingFileFromSparseFeatures(std::shared_ptr<SparseFeatures<T>> feat, float64_t* lab)
	: StreamingFileFromFeatures(feat,lab)
{
	init(feat);
}

template <class T>
StreamingFileFromSparseFeatures<T>::~StreamingFileFromSparseFeatures()
{
}

template <class T>
void StreamingFileFromSparseFeatures<T>::init(std::shared_ptr<SparseFeatures<T>> feat)
{
	features = feat;
	vector_num=0;

	set_generic<T>();
}

/* Functions to return the vector from the SparseFeatures object */
template <class T>
void StreamingFileFromSparseFeatures<T>::get_sparse_vector
(SGSparseVectorEntry<T>*& vector, int32_t& len)
{
	if (vector_num >= features->get_num_vectors())
	{
		vector=NULL;
		len=-1;
		return;
	}

	SGSparseVector<T> vec=
			features->get_sparse_feature_vector(vector_num);
	vector=vec.features;
	len=vec.num_feat_entries;

	/* TODO. check if vector needs to be freed? */

	vector_num++;
}

/* Functions to return the vector from the SparseFeatures object */
template <class T>
void StreamingFileFromSparseFeatures<T>::get_sparse_vector_and_label
(SGSparseVectorEntry<T>*& vector, int32_t& len, float64_t& label)
{
	get_sparse_vector(vector, len);
	label=labels[vector_num];
}

}
#endif //__STREAMING_FILEFROMSPARSE_H__
