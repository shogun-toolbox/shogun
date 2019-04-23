/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Vladislav Horbatiuk,
 *          Yuyu Zhang, Viktor Gal, Sergey Lisitsyn
 */
#ifndef _STREAMINGDENSEFEATURES__H__
#define _STREAMINGDENSEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/lib/DataType.h>
#include <shogun/io/streaming/InputParser.h>

namespace shogun
{
/** @brief This class implements streaming features with dense feature vectors.
 *
 * The current example is stored as a combination of current_vector
 * and current_label. Call get_next_example() followed by get_current_vector()
 * to iterate through the stream.
 */
template<class T> class StreamingDenseFeatures:
	public StreamingDotFeatures
{
public:

	/**
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CStreamingFile::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	StreamingDenseFeatures();

	/**
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 *
	 * @param file StreamingFile object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	StreamingDenseFeatures(std::shared_ptr<StreamingFile> file, bool is_labelled,
			int32_t size);

	/**
	 * Constructor taking a DenseFeatures object and a labels array
	 * as args.
	 *
	 * @param dense_features DenseFeatures object of same type
	 * @param lab labels array, float64_t*
	 */
	StreamingDenseFeatures(std::shared_ptr<DenseFeatures<T>> dense_features, float64_t* lab=
			NULL);

	/**
	 * Destructor.
	 *
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	~StreamingDenseFeatures();

	/**
	 * Sets the read function (in case the examples are
	 * unlabelled) to get_*_vector() from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 *
	 * The parser uses the function set by this while reading
	 * unlabelled examples.
	 */
	virtual void set_vector_reader();

	/**
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 *
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	virtual void set_vector_and_label_reader();

	/**
	 * Starts the parsing thread.
	 *
	 * To be called before trying to use any feature vectors from this object.
	 */
	virtual void start_parser();

	/**
	 * Ends the parsing thread.
	 *
	 * Waits for the thread to join.
	 */
	virtual void end_parser();

	/**
	 * Reset a file back to the first example
	 * if possible.
	 */
	virtual void reset_stream();

	/**
	 * Instructs the parser to return the next example.
	 *
	 * This example is stored as the current_example in this object.
	 *
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	virtual bool get_next_example();

	/**
	 * Return the current feature vector as an SGVector<T>.
	 *
	 * @return The vector as SGVector<T>
	 */
	SGVector<T> get_vector();

	/**
	 * Return the label of the current example as a float.
	 *
	 * Examples must be labelled, otherwise an error occurs.
	 *
	 * @return The label as a float64_t.
	 */
	virtual float64_t get_label();

	/**
	 * Release the current example, indicating to the parser that
	 * it has been processed by the learning algorithm.
	 *
	 * The parser is then free to throw away that example.
	 */
	virtual void release_example();

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const;

	/**
	 * Dot product using the current vector and another vector, passed as arg.
	 *
	 * @param vec The vector with which to calculate the dot product.
	 *
	 * @return Dot product as a float32_t
	 */
	virtual float32_t dot(SGVector<T> vec);

	/**
	 * Dot product taken with another StreamingDotFeatures object.
	 *
	 * Currently only works if it is a StreamingDenseFeatures object.
	 * It takes the dot product of the current_vectors of both objects.
	 *
	 * @param df CStreamingDotFeatures object.
	 *
	 * @return Dot product.
	 */
	virtual float32_t dot(std::shared_ptr<StreamingDotFeatures >df);

	/**
	 * Dot product with another dense vector.
	 *
	 * @param vec2 The dense vector with which to take the dot product.
	 * @param vec2_len length of vector
	 * @return Dot product as a float32_t.
	 */
	virtual float32_t dense_dot(const float32_t* vec2, int32_t vec2_len);

	/**
	 * Dot product with another float64_t type dense vector.
	 *
	 * @param vec2 The dense vector with which to take the dot product.
	 * @param vec2_len length of vector
	 * @return Dot product as a float64_t.
	 */
	virtual float64_t dense_dot(const float64_t* vec2, int32_t vec2_len);

	/**
	 * Add alpha*current_vector to another dense vector.
	 * Takes the absolute value of current_vector if specified.
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val=false);

	/**
	 * Add alpha*current_vector to another float64_t type dense vector.
	 * Takes the absolute value of current_vector if specified.
	 *
	 * @param alpha alpha
	 * @param vec2 vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val true if abs of current_vector should be taken
	 */
	virtual void add_to_dense_vec(float64_t alpha, float64_t* vec2,
			int32_t vec2_len, bool abs_val=false);

	/** get number of non-zero features in vector
	 *
	 * @return number of non-zero features in vector
	 */
	virtual int32_t get_nnz_features_for_vector();

	/**
	 * Return the number of features in the current example.
	 *
	 * @return number of features as int
	 */
	int32_t get_num_features();

	/**
	 * Return the feature type, depending on T.
	 *
	 * @return Feature type as EFeatureType
	 */
	virtual EFeatureType get_feature_type() const;

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_DENSE
	 */
	virtual EFeatureClass get_feature_class() const;

	/**
	 * Return the name.
	 *
	 * @return StreamingDenseFeatures
	 */
	virtual const char* get_name() const
	{
		return "StreamingDenseFeatures";
	}

	/**
	 * Return the number of vectors stored in this object.
	 *
	 * @return 1 if current_vector exists, else 0.
	 */
	virtual int32_t get_num_vectors() const;

	/** Returns a new CDebseFeatures instance which contains num_elements elements
	 * from the underlying stream. The object is not SG_REF'ed.
	 *
	 * @param num_elements num elements to save from stream
	 * @return Features object of underlying type, might contain less data if
	 * the stream did end (warning is written)
	 */
	virtual std::shared_ptr<Features> get_streamed_features(index_t num_elements);

private:
	/**
	 * Initializes members to null values.
	 * current_length is set to -1.
	 */
	void init();

	/**
	 * Calls init, and also initializes the parser with the given args.
	 *
	 * @param file StreamingFile to read from
	 * @param is_labelled whether labelled or not
	 * @param size number of examples in the parser's ring
	 */
	void init(std::shared_ptr<StreamingFile >file, bool is_labelled, int32_t size);

protected:

	/// feature weighting in combined dot features
	float32_t combined_weight;

	/// The parser object, which reads from input and returns parsed example objects.
	InputParser<T> parser;

	/// The current example's feature vector as an SGVector<T>
	SGVector<T> current_vector;

	/// The current example's label.
	float64_t current_label;
};
}
#endif // _STREAMINGDENSEFEATURES__H__
