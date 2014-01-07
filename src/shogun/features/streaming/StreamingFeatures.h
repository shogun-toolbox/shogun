/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef _STREAMING_FEATURES__H__
#define _STREAMING_FEATURES__H__

#include <lib/common.h>
#include <features/Features.h>
#include <io/streaming/StreamingFile.h>

namespace shogun
{
/** @brief Streaming features are features which are used for online
 * algorithms.
 *
 * Reading/parsing of input, and learning through the algorithm are
 * carried out in separate threads. Input is from a CStreamingFile
 * object.
 *
 * A StreamingFeatures object usually stores only one example at a
 * time, and any function like dot(), add_to_dense_vec() etc. apply
 * with this example as one implicit operand.
 *
 * Similarly, when we refer to the "feature vector" of a
 * StreamingFeatures object, it refers to the vector of the example
 * currently stored in that object.
 *
 * It is up to the user to indicate when he is done using the example
 * so that the next one can be fetched and stored in its place.
 *
 * Example objects are fetched one-by-one through a CInputParser
 * object, and therefore a StreamingFeatures object must implement the
 * following methods in the derived class:
 *
 * - start_parser(): a function to begin processing of the input.
 *
 * - end_parser(): end the parser; wait for the parsing thread to finish.
 *
 * - get_next_example(): instruct the parser to get the next example.
 *
 * - release_example(): instruct the parser to release the current example.
 *
 * - get_label(): returns the label (if applicable) for the current example.
 *
 * - get_num_features(): returns the number of features for the current example.
 *
 * - release_example() must be called before get_next_example().
 *
 * - get_streamed_features() to retreive a non-streaming instance of a certain size
 *   (has to be implemented in subclasses)
 *
 * - from_non_streaming() to stream features from an existing features object
 *   (has to be implemented in subclasses)
 *
 *   The feature vector itself may be returned through a derived class
 *   since at the moment the parser is templated for each data type.
 *
 *   Thus, a templated or specialized version of
 *   get_vector(SGVector<T>) must be implemented in the derived class.
 *
 */

class CStreamingFeatures : public CFeatures
{

public:

	/**
	 * Default constructor with no args.
	 * Doesn't do anything yet.
	 */
	CStreamingFeatures();

	/**
	 * Destructor
	 */
	virtual ~CStreamingFeatures();

	/**
	 * Set the vector reading functions.
	 *
	 * The functions are implemented specific to the type in the
	 * derived class.
	 */
	void set_read_functions();

	/**
	 * The derived object must set the function which will be used
	 * for reading one vector from the file.  This function should
	 * be a member of the CStreamingFile class.
	 *
	 * See the implementation in StreamingDenseFeatures for
	 * details.
	 */
	virtual void set_vector_reader()=0;

	/**
	 * The derived object must set the function which will be used
	 * by the parser for reading one vector and label from the
	 * file.  This function should be a member of the
	 * CStreamingFile class.
	 *
	 * See the implementation in StreamingDenseFeatures for
	 * details.
	 */
	virtual void set_vector_and_label_reader()=0;

	/**
	 * Start the parser.
	 * It stores parsed examples from the input in a separate thread.
	 */
	virtual void start_parser()=0;

	/**
	 * End the parser. Wait for the parsing thread to complete.
	 */
	virtual void end_parser()=0;

	/**
	 * Return the label of the current example.
	 *
	 * Raise an error if the input has been specified as unlabelled.
	 *
	 * @return Label (if labelled example)
	 */
	virtual float64_t get_label()=0;

	/**
	 * Indicate to the parser that it must fetch the next example.
	 *
	 * @return true on success, false on failure (i.e., no more examples).
	 */
	virtual bool get_next_example()=0;

	/**
	 * Indicate that processing of the current example is done.
	 * The parser then considers it safe to dispose of that example
	 * and replace it with another one.
	 */
	virtual void release_example()=0;

	/**
	 * Get the number of features in the current example.
	 *
	 * @return number of features in current example
	 */
	virtual int32_t get_num_features()=0;

	/**
	 * Return whether the examples are labelled or not.
	 *
	 * @return true if labelled, else false
	 */
	virtual bool get_has_labels();

	/**
	 * Whether the stream is seekable (to check if multiple epochs
	 * are possible), i.e., whether we can process examples in a
	 * batch fashion.
	 *
	 * A stream can usually seekable when it comes from a file or
	 * when it comes from another conventional CFeatures object.
	 *
	 * @return true if seekable, else false.
	 */
	virtual bool is_seekable();

	/**
	 * Function to reset the stream (if possible).
	 */
	virtual void reset_stream();

	/** Returns a CFeatures instance which contains num_elements elements from
	 * the underlying stream
	 *
	 * @param num_elements num elements to save from stream
	 * @return CFeatures object of underlying type, NULL if not enough data
	 *
	 * NOT IMPLEMENTED!
	 */
	virtual CFeatures* get_streamed_features(index_t num_elements)
	{
		SG_ERROR("%s::get_streamed_features() is not yet implemented!\n",
				get_name());
		return NULL;
	}

protected:

	/// Whether examples are labelled or not.
	bool has_labels;

	/// The StreamingFile object to read from.
	CStreamingFile* working_file;

	/// Whether the stream is seekable
	bool seekable;

};
}
#endif // _STREAMING_FEATURES__H__
