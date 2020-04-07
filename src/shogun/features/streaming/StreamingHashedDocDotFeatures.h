/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Evangelos Anagnostopoulos, Yuyu Zhang, Bjoern Esser, Viktor Gal
 */
#ifndef _STREAMING_HASHEDDOCDOTFEATURES__H__
#define _STREAMING_HASHEDDOCDOTFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/StringFeatures.h>
#include <shogun/features/streaming/StreamingDotFeatures.h>
#include <shogun/lib/Tokenizer.h>
#include <shogun/converter/HashedDocConverter.h>
#include <shogun/io/streaming/InputParser.h>
#include <shogun/io/streaming/StreamingFileFromStringFeatures.h>

namespace shogun
{
class StreamingDotFeatures;
class Tokenizer;
class HashedDocConverter;

/** @brief This class implements streaming features for a document collection.
 * Like in the standard Bag-of-Words representation, this class considers each document as a collection of tokens,
 * which are then hashed into a new feature space of a specified dimension.
 * This class is very flexible and allows the user to specify the tokenizer used to tokenize each document,
 * specify whether the results should be normalized with regards to the sqrt of the document size, as well
 * as to specify whether he wants to combine different tokens.
 * The latter implements a k-skip n-grams approach, meaning that you can combine up to n tokens, while skipping up to k.
 * Eg. for the tokens ["a", "b", "c", "d"], with n_grams = 2 and skips = 2, one would get the following combinations :
 * ["a", "ab", "ac" (skipped 1), "ad" (skipped 2), "b", "bc", "bd" (skipped 1), "c", "cd", "d"].
 *
 * The current example is stored as a combination of current_vector
 * and current_label. Call get_next_example() followed by get_current_vector()
 * to iterate through the stream.
 */
class StreamingHashedDocDotFeatures : public StreamingDotFeatures
{
public:
	/** Constructor */
	StreamingHashedDocDotFeatures();

	/**
	 * Constructor with input information passed.
	 * Will use normalization and no quadratic features by default, user should
	 * use the set_normalization() and set_k_skip_n_gram() methods to change that.
	 *
	 * @param file CStreamingFile to take input from.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of examples to be held in the parser's "ring".
	 * @param tzer the tokenizer to use on the document collection
	 * @param bits the number of bits of the new dimension (means a dimension of size 2^bits)
	 */
	StreamingHashedDocDotFeatures(std::shared_ptr<StreamingFile> file, bool is_labelled, int32_t size,
			std::shared_ptr<Tokenizer> tzer, int32_t bits=20);

	/**
	 * Constructor taking a DotFeatures object and optionally,
	 * labels, as args.
	 * Will use normalization and no quadratic features by default, user should
	 * use the set_normalization() and set_k_skip_n_gram() methods to change that.
	 *
	 * The derived class should implement it so that the
	 * Streaming*Features class uses the DotFeatures object as the
	 * input, getting examples one by one from the DotFeatures
	 * object (and labels, if applicable).
	 *
	 * @param dot_features DotFeatures object
	 * @param tzer the tokenizer to use on the document collection
	 * @param bits the number of bits of the new dimension (means a dimension of size 2^bits)
	 * @param lab labels (optional)
	 */
	StreamingHashedDocDotFeatures(std::shared_ptr<StringFeatures<char>> dot_features,std::shared_ptr<Tokenizer> tzer,
			int32_t bits=20, float64_t* lab=NULL);

	/** Destructor */
	~StreamingHashedDocDotFeatures() override;

	/** compute dot product between vectors of two
	 * StreamingDotFeatures objects.
	 *
	 * @param df StreamingDotFeatures (of same kind) to compute
	 * dot product with
	 */
	float32_t dot(std::shared_ptr<StreamingDotFeatures> df) override;

	/** compute dot product between current vector and a dense vector
	 *
	 * @param vec2 real valued vector
	 * @param vec2_len length of vector
	 */
	float32_t dense_dot(const float32_t* vec2, int32_t vec2_len) override;

	/** add current vector multiplied with alpha to dense vector, 'vec'
	 *
	 * @param alpha scalar alpha
	 * @param vec2 real valued vector to add to
	 * @param vec2_len length of vector
	 * @param abs_val if true add the absolute value
	 */
	void add_to_dense_vec(float32_t alpha, float32_t* vec2,
			int32_t vec2_len, bool abs_val=false) override;

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	int32_t get_dim_feature_space() const override;

	/**
	 * Return the name.
	 *
	 * @return the name of the class
	 */
	const char* get_name() const override;

	/**
	 * Return the number of vectors stored in this object.
	 *
	 * @return 1 if current_vector exists, else 0.
	 */
	int32_t get_num_vectors() const override;

	/**
	 * Sets the read function (in case the examples are
	 * unlabelled) to get_*_vector() from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 *
	 * The parser uses the function set by this while reading
	 * unlabelled examples.
	 */
	void set_vector_reader() override;

	/**
	 * Sets the read function (in case the examples are labelled)
	 * to get_*_vector_and_label from CStreamingFile.
	 *
	 * The exact function depends on type T.
	 *
	 * The parser uses the function set by this while reading
	 * labelled examples.
	 */
	void set_vector_and_label_reader() override;

	/**
	 * Return the feature type, depending on T.
	 *
	 * @return Feature type as EFeatureType
	 */
	EFeatureType get_feature_type() const override;

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_DENSE
	 */
	EFeatureClass get_feature_class() const override;

	/**
	 * Start the parser.
	 * It stores parsed examples from the input in a separate thread.
	 */
	void start_parser() override;

	/**
	 * End the parser. Wait for the parsing thread to complete.
	 */
	void end_parser() override;

	/**
	 * Return the label of the current example.
	 *
	 * Raise an error if the input has been specified as unlabelled.
	 *
	 * @return Label (if labelled example)
	 */
	float64_t get_label() override;

	/**
	 * Indicate to the parser that it must fetch the next example.
	 *
	 * @return true on success, false on failure (i.e., no more examples).
	 */
	bool get_next_example() override;

	/**
	 * Indicate that processing of the current example is done.
	 * The parser then considers it safe to dispose of that example
	 * and replace it with another one.
	 */
	void release_example() override;

	/**
	 * Get the number of features in the current example.
	 *
	 * @return number of features in current example
	 */
	int32_t get_num_features() override;

	/** Get the current example
	 *
	 * @return a SGSparseVector representing the hashed version of the string last read
	 */
	SGSparseVector<float64_t> get_vector();

	/** specify whether hashed vector should be normalized or not
	 *
	 * @param normalize  whether to normalize
	 */
	void set_normalization(bool normalize);

	/** Method used to specify the parameters for the quadratic
	 * approach of k-skip n-grams. See class description for more
	 * details and an example.
	 *
	 * @param k the max number of allowed skips
	 * @param n the max number of tokens to combine
	 */
	void set_k_skip_n_grams(int32_t k, int32_t n);

private:
	void init(const std::shared_ptr<StreamingFile>& file, bool is_labelled, int32_t size, std::shared_ptr<Tokenizer> tzer,
		int32_t bits, bool normalize, int32_t n_grams, int32_t skips);

protected:

	/** number of bits for the target dimension */
	int32_t num_bits;

	/** Current example */
	SGSparseVector<float64_t> current_vector;

	/** Tokenizer */
	std::shared_ptr<Tokenizer >tokenizer;

	/** Converter */
	std::shared_ptr<HashedDocConverter> converter;

	/** The parser */
	InputParser<char> parser;

	/** The current example's label */
	float64_t current_label;
};
}

#endif // _STREAMING_HASHEDDOCDOTFEATURES__H__
