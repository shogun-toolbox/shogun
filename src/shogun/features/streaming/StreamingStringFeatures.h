/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Vladislav Horbatiuk, Yuyu Zhang, Viktor Gal, Thoralf Klein, 
 *          Sergey Lisitsyn, Soeren Sonnenburg, Wu Lin
 */
#ifndef _STREAMING_STRINGFEATURES__H__
#define _STREAMING_STRINGFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/DataType.h>
#include <shogun/io/streaming/InputParser.h>

#include <shogun/lib/SGVector.h>
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/Alphabet.h>

namespace shogun
{
/** @brief This class implements streaming features as strings.
 *
 */
template <class T> class StreamingStringFeatures : public StreamingFeatures
{
public:

	/**
	 * Default constructor.
	 *
	 * Sets the reading functions to be
	 * CStreamingFile::get_*_vector and get_*_vector_and_label
	 * depending on the type T.
	 */
	StreamingStringFeatures();

	/**
	 * Constructor taking args.
	 * Initializes the parser with the given args.
	 *
	 * @param file StreamingFile object, input file.
	 * @param is_labelled Whether examples are labelled or not.
	 * @param size Number of example objects to be stored in the parser at a time.
	 */
	StreamingStringFeatures(std::shared_ptr<StreamingFile> file,
				 bool is_labelled,
				 int32_t size);

	/**
	 * Destructor.
	 *
	 * Ends the parsing thread. (Waits for pthread_join to complete)
	 */
	~StreamingStringFeatures() override;

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
	 * Set the alphabet to be used.
	 * Call before parsing.
	 *
	 * @param alpha alphabet as an EAlphabet enum.
	 */
	void use_alphabet(EAlphabet alpha);

	/**
	 * Set the alphabet to be used.
	 * Call before parsing.
	 *
	 * @param alpha alphabet as a pointer to a Alphabet object.
	 */
	void use_alphabet(std::shared_ptr<Alphabet> alpha);

	/**
	 * Set whether remapping to another alphabet is required.
	 *
	 * Call before parsing.
	 * @param ascii_alphabet the alphabet to convert from, Alphabet*
	 * @param binary_alphabet the alphabet to convert to, Alphabet*
	 */
	void set_remap(std::shared_ptr<Alphabet> ascii_alphabet, std::shared_ptr<Alphabet> binary_alphabet);

	/**
	 * Set whether remapping to another alphabet is required.
	 *
	 * Call before parsing.
	 * @param ascii_alphabet the alphabet to convert from, EAlphabet
	 * @param binary_alphabet the alphabet to convert to, EAlphabet
	 */
	void set_remap(EAlphabet ascii_alphabet=DNA, EAlphabet binary_alphabet=RAWDNA);

	/**
	 * Return the alphabet being used as a Alphabet*
	 * @return
	 */
	std::shared_ptr<Alphabet> get_alphabet();

	/** get number of symbols
	 *
	 * Note: floatmax_t sounds weird, but LONG is not long enough
	 *
	 * @return number of symbols
	 */
	floatmax_t get_num_symbols();

	/**
	 * Starts the parsing thread.
	 *
	 * To be called before trying to use any feature vectors from this object.
	 */
	void start_parser() override;

	/**
	 * Ends the parsing thread.
	 *
	 * Waits for the thread to join.
	 */
	void end_parser() override;

	/**
	 * Instructs the parser to return the next example.
	 *
	 * This example is stored as the current_example in this object.
	 *
	 * @return True on success, false if there are no more
	 * examples, or an error occurred.
	 */
	bool get_next_example() override;

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
	float64_t get_label() override;

	/**
	 * Release the current example, indicating to the parser that
	 * it has been processed by the learning algorithm.
	 *
	 * The parser is then free to throw away that example.
	 */
	void release_example() override;

	/**
	 * Return the length of the current vector.
	 *
	 * @return current vector length as int32_t
	 */
	virtual int32_t get_vector_length();

	/**
	 * Return the feature type, depending on T.
	 *
	 * @return Feature type as EFeatureType
	 */
	EFeatureType get_feature_type() const override;

	/**
	 * Return the feature class
	 *
	 * @return C_STREAMING_STRING
	 */
	EFeatureClass get_feature_class() const override;

	/**
	 * Return the name.
	 *
	 * @return StreamingSparseFeatures
	 */
	const char* get_name() const override { return "StreamingStringFeatures"; }

	/**
	 * Return the number of vectors stored in this object.
	 *
	 * @return 1 if current_vector exists, else 0.
	 */
	int32_t get_num_vectors() const override;

	/**
	 * Return the number of features in the current vector.
	 *
	 * @return length of the vector
	 */
	int32_t get_num_features() override;

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

	/// The parser object, which reads from input and returns parsed example objects.
	InputParser<T> parser;

	/// Alphabet to use
	std::shared_ptr<Alphabet> alphabet;

	/// If remapping is enabled, this is the source alphabet
	std::shared_ptr<Alphabet> alpha_ascii;

	/// If remapping is enabled, this is the target alphabet
	std::shared_ptr<Alphabet> alpha_bin;

	/// The current example's string
	SGVector<T> current_string;

	/// The label of the current example, if applicable
	float64_t current_label;

	/// Whether examples are labelled or not
	bool has_labels;

	/// Whether remapping must be done
	bool remap_to_bin;

	/// Number of symbols
	int32_t num_symbols;
};

}
#endif // _STREAMING_STRINGFEATURES__H__
