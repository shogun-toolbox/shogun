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
#include "lib/buffer.h"
#include <pthread.h>

namespace shogun
{
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
		 * @param size Size of the buffer in MB
		 */
		void init(CStreamingFile* input_file, bool is_labelled, int32_t size);

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
		 * @param ex Example to be copied.
		 */
		void copy_example_into_buffer(example* ex);

		/** 
		 * Retrieves the next example from the buffer.
		 * 
		 * 
		 * @return The example pointer.
		 */
		example* retrieve_example();
		
		/**
		 * Gets the next example, assuming it to be labelled.
		 *
		 * Waits till retrieve_example returns a valid example, or
		 * returns if reading is done already.
		 *
		 * @param feature_vector Feature vector pointer
		 * @param length Length of feature vector
		 * @param label Label of example
		 *
		 * @return 1 if an example could be fetched, 0 otherwise
		 */
		int32_t get_next_example(float64_t* &feature_vector,
										  int32_t &length,
										  float64_t &label);

		/** 
		 * Gets the next example, assuming it to be unlabelled.
		 * 
		 * @param feature_vector 
		 * @param length 
		 * 
		 * @return 1 if an example could be fetched, 0 otherwise
		 */
		int32_t get_next_example(float64_t* &feature_vector,
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

		ParseBuffer* examples_buff;
		
		int32_t number_of_features;
		int32_t number_of_vectors_parsed;
		int32_t number_of_vectors_read;

		example* current_example;
		
		SGVector<float64_t> current_fv;	/**< Yet to be used in the code! */
		float64_t* current_feature_vector; /**< Points to feature
											* vector of last read example */
		
		float64_t current_label; /**< Label of last read example */
		
		int32_t current_len; /**< Features in last
							  * read example */

		
	};
}
#endif // __INPUTPARSER_H__
