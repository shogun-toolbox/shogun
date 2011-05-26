/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#ifndef __INPUTPARSER_H__
#define __INPUTPARSER_H__

#include "lib/io.h"
#include "lib/StreamingFile.h"
#include "lib/common.h"
#include <pthread.h>

namespace shogun
{
	/// Unlabelled example class
	class UnlabelledExample
	{
	public:
		float64_t* feature_vector;
		int32_t dimensions;
	};

	/// Labelled example class
	class LabelledExample
	{
	public:
		float64_t* feature_vector;
		int32_t dimensions;
		float64_t label;
	};

	enum E_IS_EXAMPLE_USED
	{
		E_EMPTY = 1,
		E_NOT_USED = 2,
		E_USED = 3
	};

	enum E_EXAMPLE_TYPE
	{
		E_LABELLED = 1,
		E_UNLABELLED = 2
	};

	class CInputParser
	{
	public:

		/**
		 * Constructor
		 *
		 */
		CInputParser();

		/**
		 * Destructor
		 *
		 */
		~CInputParser();

		/**
		 * Initializer
		 *
		 * Sets initial or default values for members.
		 * is_example_used is initialized to EMPTY.
		 * example_type is LABELLED by default.
		 *
		 * @param input_file CStreamingFile object
		 * @param is_labelled Whether example is labelled or not (bool), optional
		 */
		void init(CStreamingFile* input_file, bool is_labelled);

		/**
		 * Test if parser is running.
		 *
		 * @return true if running, false otherwise.
		 */
		bool is_running();

		/**
		 * Get number of features from example.
		 * Currently reads first line of input to infer.
		 *
		 * @return Number of features
		 */
		int32_t get_number_of_features();

		/**
		 * Gets feature vector, length and label.
		 * Sets their values by reference.
		 * Uses method for reading the vector defined in CStreamingFile.
		 *
		 * @param feature_vector Pointer to feature vector
		 * @param length Features in vector
		 * @param label Label of example
		 *
		 * @return 1 on success, 0 on failure.
		 */
		int32_t get_vector_and_label(float64_t* &feature_vector,
									 int32_t &length,
									 float64_t &label);

		/**
		 * Gets feature vector and length by reference.
		 * Assumes examples are unlabelled.
		 * Uses method for reading the vector defined in CStreamingFile.
		 *
		 * @param feature_vector Pointer to feature vector
		 * @param length Features in vector
		 *
		 * @return 1 on success, 0 on failure
		 */
		int32_t get_vector_only(float64_t* &feature_vector, int32_t &length);

		/**
		 * Starts the parser, creating a new thread.
		 *
		 * main_parse_loop is the parsing method.
		 */
		void start_parser();

		/**
		 * Set buffer size in units of number of examples.
		 *
		 * @param size Size of buffer
		 */
		void set_buffer_size(int32_t size);

		/**
		 * Increment buffer write index.
		 *
		 */
		void buffer_increment_write_index();

		/**
		 * Increment buffer read index.
		 *
		 */
		void buffer_increment_read_index();

		/**
		 * Main parsing loop. Reads examples from source and stores
		 * them in the buffer.
		 *
		 * @param params 'this' object
		 *
		 * @return NULL
		 */
		void* main_parse_loop(void* params);

		/**
		 * Copy example into the buffer.
		 *
		 * Buffer space:
		 * -Example n-
		 * float64_t* feature_vector
		 * int32_t length
		 * float64_t label (if LabelledExample)
		 * list of float64_ts (Feature vector)
		 * -Example n+1-
		 * ..
		 *
		 * Currently, constant dimensionality is assumed!
		 *
		 * @param example Example to be copied.
		 */
		void copy_example_into_buffer(void* example);

		/**
		 * Gets the next unused example from the buffer.
		 *
		 * @return NULL if no unused example is available, else return
		 * a pointer to the example in the buffer.
		 */
		void* get_next_example();

		/**
		 * Gets the next example, assuming it to be labelled.
		 *
		 * Waits till get_next_example returns a valid example, or
		 * returns if reading is done already.
		 *
		 * @param feature_vector Feature vector pointer
		 * @param length Length of feature vector
		 * @param label Label of example
		 *
		 * @return 1 if an example could be fetched, 0 otherwise
		 */
		int32_t get_next_example_labelled(float64_t* &feature_vector,
										  int32_t &length,
										  float64_t &label);

		/**
		 * Gets the next example, assuming it to be unlabelled.
		 *
		 * Waits till get_next_example returns a valid example, or
		 * returns if reading is done already.
		 *
		 * @param feature_vector Feature vector pointer
		 * @param length Length of feature vector
		 *
		 * @return 1 if an example could be fetched, 0 otherwise
		 */
		int32_t get_next_example_unlabelled(float64_t* &feature_vector,
											int32_t &length);

		/**
		 * Finalize the current example, indicating that the buffer
		 * position it occupies may be overwritten by the parser.
		 *
		 * Should be called when the example has been processed by the
		 * external algorithm.
		 */
		void finalize_example();

		/**
		 * End the parser, closing the parse thread.
		 *
		 */
		void end_parser();

	private:
		/**
		 * Entry point for the parse thread.
		 *
		 * @param params this object
		 *
		 * @return NULL
		 */
		static void* parse_loop_entry_point(void* params);


	public:
		bool parsing_done;	/**< true if all input is parsed */
		bool reading_done;	/**< true if all examples are fetched */

		E_EXAMPLE_TYPE example_type; /**< LABELLED or UNLABELLED */

	protected:

		CStreamingFile* input_source; /**< Input source,
									   * CStreamingFile object */

		pthread_t parse_thread;/**< Parse thread */

		void* examples_buff;	/**< Buffer for examples, behaves
								 * like a ring.
								 * Examples are stored and retrieved
								 * from this buffer.*/

		int32_t buffer_write_index; /**< Where next example will be
									 * written into the buffer. */

		int32_t buffer_read_index; /**< Where next example will be
									* read from the buffer */

		E_IS_EXAMPLE_USED* is_example_used; /**< Indicates state of examples
											 * in buffer - used, not
											 * used, or empty. */

		pthread_cond_t* example_in_use_condition;
		pthread_mutex_t* example_in_use_mutex;

		int32_t example_memsize; /**< Size of example object */
		int32_t buffer_size;	/**< Number of examples to store in buffer */

		int32_t number_of_features;

		int32_t number_of_vectors_parsed;
		int32_t number_of_vectors_read;

		float64_t* current_feature_vector; /**< Points to feature
											* vector of last read example */
		
		float64_t current_label; /**< Label of last read example */
		
		int32_t current_number_of_features; /**< Features in last
											 * read example */
		
		void* current_example;	/**< Points to current example in buffer */



	};
}
#endif // __INPUTPARSER_H__
