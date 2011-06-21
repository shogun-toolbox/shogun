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
#include "lib/ParseBuffer.h"
#include <pthread.h>

#define PARSER_DEFAULT_BUFFSIZE 100

namespace shogun
{
	enum E_EXAMPLE_TYPE
	{
		E_LABELLED = 1,
		E_UNLABELLED = 2
	};

	template <class T>
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
		 * @param size Size of the buffer in number of examples
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
		 * Reads a line from the input and stores it as a vector of type T.
		 * 
		 * @param feature_vector Vector pointer, which is malloced and set by reference.
		 * @param length Length of the vector returned
		 */
		void read_vector(T* &feature_vector,
				 int32_t &length);				

		/** 
		 * Reads a line from the input, taking the first element to be
		 * the label, and the rest as a vector of type T.
		 * 
		 * @param feature_vector Vector pointer, malloced and set by reference.
		 * @param length Length of the vector returned
		 * @param label Label of the example
		 */
		void read_vector_and_label(T* &feature_vector,
					   int32_t &length,
					   float64_t &label);
		
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
		int32_t get_vector_and_label(T* &feature_vector,
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
		int32_t get_vector_only(T* &feature_vector, int32_t &length);

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
		void copy_example_into_buffer(example<T>* ex);

		/** 
		 * Retrieves the next example from the buffer.
		 * 
		 * 
		 * @return The example pointer.
		 */
		example<T>* retrieve_example();
		
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
		int32_t get_next_example(T* &feature_vector,
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
		int32_t get_next_example(T* &feature_vector,
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

		CParseBuffer<T>* examples_buff;
		
		int32_t number_of_features;
		int32_t number_of_vectors_parsed;
		int32_t number_of_vectors_read;

		example<T>* current_example;
		
		SGVector<T> current_fv;	/**< Yet to be used in the code! */
		T* current_feature_vector; /**< Points to feature
					    * vector of last read example */
		
		float64_t current_label; /**< Label of last read example */
		
		int32_t current_len; /**< Features in last
				      * read example */
		
	};

#define READ_VECTOR(sg_type, sg_function)						\
	template <> void CInputParser<sg_type>::read_vector(sg_type* &vec, int32_t &len) \
	{								\
		input_source->sg_function(vec, len);		\
	}

	READ_VECTOR(bool, get_bool_vector);
	READ_VECTOR(char, get_char_vector);
	READ_VECTOR(int8_t, get_int8_vector);
	READ_VECTOR(uint8_t, get_byte_vector);
	READ_VECTOR(int16_t, get_short_vector);
	READ_VECTOR(uint16_t, get_word_vector);
	READ_VECTOR(int32_t, get_int_vector);
	READ_VECTOR(uint32_t, get_uint_vector);
	READ_VECTOR(int64_t, get_long_vector);
	READ_VECTOR(uint64_t, get_ulong_vector);
	READ_VECTOR(float32_t, get_shortreal_vector);
	READ_VECTOR(float64_t, get_real_vector);
	READ_VECTOR(floatmax_t, get_longreal_vector);
	
#undef READ_VECTOR_AND_LABEL		

	#define READ_VECTOR_AND_LABEL(sg_type, sg_function)						\
		template <> void CInputParser<sg_type>::read_vector_and_label(sg_type* &vec, int32_t &len, float64_t &label) \
	{								\
		input_source->sg_function(vec, len, label);		\
	}

	READ_VECTOR_AND_LABEL(bool, get_bool_vector_and_label);
	READ_VECTOR_AND_LABEL(char, get_char_vector_and_label);
	READ_VECTOR_AND_LABEL(int8_t, get_int8_vector_and_label);
	READ_VECTOR_AND_LABEL(uint8_t, get_byte_vector_and_label);
	READ_VECTOR_AND_LABEL(int16_t, get_short_vector_and_label);
	READ_VECTOR_AND_LABEL(uint16_t, get_word_vector_and_label);
	READ_VECTOR_AND_LABEL(int32_t, get_int_vector_and_label);
	READ_VECTOR_AND_LABEL(uint32_t, get_uint_vector_and_label);
	READ_VECTOR_AND_LABEL(int64_t, get_long_vector_and_label);
	READ_VECTOR_AND_LABEL(uint64_t, get_ulong_vector_and_label);
	READ_VECTOR_AND_LABEL(float32_t, get_shortreal_vector_and_label);
	READ_VECTOR_AND_LABEL(float64_t, get_real_vector_and_label);
	READ_VECTOR_AND_LABEL(floatmax_t, get_longreal_vector_and_label);
	
#undef READ_VECTOR_AND_LABEL		

	template <class T>
		CInputParser<T>::CInputParser()
	{
		//init(NULL, true, PARSER_DEFAULT_BUFFSIZE);
	}

	template <class T>
		CInputParser<T>::~CInputParser()
	{
		end_parser();
	
		delete current_example;
		delete examples_buff;
	}

	template <class T>
		void CInputParser<T>::init(CStreamingFile* input_file, bool is_labelled = true, int32_t size = PARSER_DEFAULT_BUFFSIZE)
	{
		input_source = input_file;

		if (is_labelled == true)
			example_type = E_LABELLED;
		else
			example_type = E_UNLABELLED;

		examples_buff = new CParseBuffer<T>(size);
		current_example = new example<T>();
	
		parsing_done = false;
		reading_done = false;
		number_of_vectors_parsed = 0;
		number_of_vectors_read = 0;

		current_len = -1;
		current_label = -1;
		current_feature_vector = NULL;
	}

	template <class T>
		void CInputParser<T>::start_parser()
	{
		pthread_create(&parse_thread, NULL, parse_loop_entry_point, this);
	}

	template <class T>
		void* CInputParser<T>::parse_loop_entry_point(void* params)
	{
		((CInputParser *) params)->main_parse_loop(params);

		return NULL;
	}

	template <class T>
		bool CInputParser<T>::is_running()
	{
		if (parsing_done)
			if (reading_done)
				return false;
			else
				return true;
		else
			return false;
	}

	template <class T>
		int32_t CInputParser<T>::get_vector_and_label(T* &feature_vector,
							      int32_t &length,
							      float64_t &label)
	{
		read_vector_and_label(feature_vector, length, label);

		if (length < 1)
		{
			// Problem reading the example
			return 0;
		}

		return 1;
	}

	template <class T>
		int32_t CInputParser<T>::get_vector_only(T* &feature_vector,
							 int32_t &length)
	{
		read_vector(feature_vector, length);

		if (length < 1)
		{
			// Problem reading the example
			return 0;
		}

		return 1;
	}

	template <class T>
		void CInputParser<T>::copy_example_into_buffer(example<T>* ex)
	{
		examples_buff->copy_example(ex);
	}

	template <class T>
		void* CInputParser<T>::main_parse_loop(void* params)
	{
		// Read the examples into current_* objects
		// Instead of allocating mem for new objects each time

		CInputParser* this_obj = (CInputParser *) params;
		this->input_source = this_obj->input_source;

		while (!parsing_done)
		{
			if (example_type == E_LABELLED)
				get_vector_and_label(current_feature_vector, current_len, current_label);
			else
				get_vector_only(current_feature_vector,	current_len);

			if (current_len < 0)
			{
				parsing_done = true;
				return NULL;
			}

			current_example->label = current_label;
			current_example->fv.vector = current_feature_vector;
			current_example->fv.length = current_len;

			examples_buff->copy_example(current_example);
			number_of_vectors_parsed++;
		}

		return NULL;
	}

	template <class T>
		example<T>* CInputParser<T>::retrieve_example()
	{
		// Return the next unused example from the buffer

		example<T> *ex;
		
		if (number_of_vectors_parsed <= 0)
			return NULL;

		if (parsing_done)
		{
			if (number_of_vectors_read == number_of_vectors_parsed)
			{
				reading_done = true;
				return NULL;
			}
		}

		if (number_of_vectors_read == number_of_vectors_parsed)
		{
			return NULL;
		}

		ex = examples_buff->fetch_example();
		number_of_vectors_read++;
		
		return ex;
	}

	template <class T>
		int32_t CInputParser<T>::get_next_example(T* &fv, int32_t &length, float64_t &label)
	{
		/* if reading is done, no more examples can be fetched. return 0
		   else, if example can be read, get the example and return 1.
		   otherwise, wait for further parsing, get the example and
		   return 1 */
	
		example<T> *ex;

		while (1)
		{
			if (reading_done)
				return 0;

			ex = retrieve_example();
		
			if (ex == NULL)
				continue;
			else
				break;
		}
	
		fv = ex->fv.vector;
		length = ex->fv.length;
		label = ex->label;

		return 1;
	}

	template <class T>
		int32_t CInputParser<T>::get_next_example(T* &fv, int32_t &length)
	{
		float64_t label_dummy;
	
		return get_next_example(fv, length, label_dummy);
	}

	template <class T>
		void CInputParser<T>::finalize_example()
	{
		examples_buff->finalize_example();
	}

	template <class T>
		void CInputParser<T>::end_parser()
	{
		pthread_join(parse_thread, NULL);
	}

}
#endif // __INPUTPARSER_H__
