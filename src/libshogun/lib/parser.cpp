/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "lib/common.h"
#include "lib/io.h"
#include "lib/parser.h"

#include <stdio.h>
#include <string.h>

#define PARSER_DEFAULT_BUFFSIZE 100

using namespace shogun;

CInputParser::CInputParser()
{
	init(NULL, true);
}

CInputParser::~CInputParser()
{
	end_parser();
	if (example_type == E_LABELLED)
		delete[] (LabelledExample*) examples_buff;
	else
		delete[] (UnlabelledExample*) examples_buff;
	delete[] feature_vectors_buff;
	delete[] is_example_used;
	delete[] example_in_use_condition;
	delete[] example_in_use_mutex;
}

void CInputParser::init(CStreamingFile* input_file, bool is_labelled = true)
{
	input_source = input_file;
	buffer_size = PARSER_DEFAULT_BUFFSIZE;

	if (is_labelled == true)
		example_type = E_LABELLED;
	else
		example_type = E_UNLABELLED;

	examples_buff = NULL;
	parsing_done = false;
	reading_done = false;
	is_example_used = NULL;
	number_of_vectors_parsed = 0;
	number_of_vectors_read = 0;

	current_number_of_features = -1;
	current_label = -1;
	current_feature_vector = NULL;
	buffer_write_index = 0;
	buffer_read_index = -1;

	is_example_used = new E_IS_EXAMPLE_USED[buffer_size];
	example_in_use_condition = new pthread_cond_t[buffer_size];
	example_in_use_mutex = new pthread_mutex_t[buffer_size];

	for (int i=0; i<buffer_size; i++)
	{
		is_example_used[i] = E_EMPTY;
		pthread_cond_init(&example_in_use_condition[i], NULL);
		pthread_mutex_init(&example_in_use_mutex[i], NULL);
	}


}

void CInputParser::start_parser()
{
	pthread_create(&parse_thread, NULL, parse_loop_entry_point, this);
}

void* CInputParser::parse_loop_entry_point(void* params)
{
	((CInputParser *) params)->main_parse_loop(params);

	return NULL;
}

bool CInputParser::is_running()
{
    if (parsing_done)
		if (reading_done)
			return false;
		else
			return true;
    else
		return false;
}

int32_t CInputParser::get_number_of_features()
{
	// Get the number of features in the line
	// Assumes it is used only at the first line of input
	// Will fseek back to zero!

	int32_t ret, length;
	float64_t* feature_vector, label;

	if (example_type == E_LABELLED)
	{
		ret = get_vector_and_label(feature_vector, length, label);
	}

	else
	{
		ret = get_vector_only(feature_vector, length);
	}

	if (!ret)
		return -1;	// No examples could be read

	input_source->seek_to_zero(); // Seek back to zero for further processing

	return length;
}

int32_t CInputParser::get_vector_and_label(float64_t* &feature_vector,
										   int32_t &length,
										   float64_t &label)
{
	input_source->get_real_vector(feature_vector, length);
	/* The get_real_vector call should be replaced with
	   a dynamic call depending on the type of feature. */

	if (length < 2)
	{
		parsing_done=true;
		return 0;	// Problem reading the example
	}

	label=feature_vector[0];
	feature_vector++;
	length--;

	return 1;
}

int32_t CInputParser::get_vector_only(float64_t* &feature_vector,
									  int32_t &length)
{
	input_source->get_real_vector(feature_vector, length);
	/* The get_real_vector call should be replaced with
	   a dynamic call depending on the type of feature. */

	if (length < 1)
	{
		parsing_done=true;
		return 0;	// Problem reading the example
	}

	return 1;
}

void CInputParser::buffer_increment_write_index()
{
	buffer_write_index = (buffer_write_index + 1) % buffer_size;
}

void CInputParser::buffer_increment_read_index()
{
	buffer_read_index = (buffer_read_index + 1) % buffer_size;
}

void CInputParser::copy_example_into_buffer(void* example)
{
	/* First we should check if the example can be overwritten or
	   not. In case the same buffer space is being used wait on a
	   cond to be true, else lock the code here */


	/* if the ex. is not used, then lock.

	   if already used, it is safe to overwrite the location */

	void* current_example_loc;

	if (is_example_used[buffer_write_index] == E_NOT_USED)
		pthread_mutex_lock(&example_in_use_mutex[buffer_write_index]);

	while (is_example_used[buffer_write_index] == E_NOT_USED)
	{
		pthread_cond_wait(&example_in_use_condition[buffer_write_index], &example_in_use_mutex[buffer_write_index]);
	}

	// Find where to store the example in the buffer

	if (example_type == E_LABELLED)
	{
		current_example_loc = ((char *)examples_buff + buffer_write_index*example_memsize);

		((LabelledExample *) current_example_loc)->dimensions = ((LabelledExample *) example)->dimensions;

		((LabelledExample *) current_example_loc)->label = ((LabelledExample *) example)->label;

		for (int i=0;i<((LabelledExample *) current_example_loc)->dimensions; i++)
		{
			feature_vectors_buff[buffer_write_index*number_of_features + i] = ((LabelledExample* ) example)->feature_vector[i];
		}

		((LabelledExample *) current_example_loc)->feature_vector = &feature_vectors_buff[buffer_write_index*number_of_features];
		
	}

	else
	{
		current_example_loc = ((char *)examples_buff + buffer_write_index*example_memsize);

		((UnlabelledExample *) current_example_loc)->dimensions = ((UnlabelledExample *) example)->dimensions;

		for (int i=0; i<((UnlabelledExample *) current_example_loc)->dimensions; i++)
		{
			feature_vectors_buff[buffer_write_index*number_of_features + i] = ((UnlabelledExample *) example)->feature_vector[i];
		}
		
		((UnlabelledExample *) current_example_loc)->feature_vector = &feature_vectors_buff[buffer_write_index*number_of_features];
		
	}

	is_example_used[buffer_write_index] = E_NOT_USED; // set the example to unused
	pthread_mutex_unlock(&example_in_use_mutex[buffer_write_index]);
}


void* CInputParser::main_parse_loop(void* params)
{
	// Read the examples into current_* objects
	// Instead of allocating mem for new objects each time

	CInputParser* this_obj = (CInputParser *) params;
	this->input_source = this_obj->input_source;

	while (!parsing_done)
	{
		// Get number of features from the first example parsed
		if (number_of_vectors_parsed == 0)
		{
			// Get number of features, allocate mem
			current_number_of_features = get_number_of_features();
			number_of_features = current_number_of_features;

			// Now allocate mem for buffer
			if (example_type == E_LABELLED)
			{
				example_memsize = sizeof(LabelledExample);
				current_example = new LabelledExample;
				examples_buff = new LabelledExample[buffer_size];
				feature_vectors_buff = new float64_t[buffer_size*number_of_features];
			}

			else
			{
				example_memsize = sizeof(UnlabelledExample);
				current_example = new UnlabelledExample;
				examples_buff = new UnlabelledExample[buffer_size];
				feature_vectors_buff = new float64_t[buffer_size*number_of_features];
			}

			// make it point to the list of floats in current_example
			// current_feature_vector=feature_vectors_buff;
			
		}

		if (example_type == E_LABELLED)
		{

			get_vector_and_label(current_feature_vector,
								 current_number_of_features,
								 current_label);
		}

		else if (example_type == E_UNLABELLED)
		{
			get_vector_only(current_feature_vector,
							current_number_of_features);
		}

		if (current_number_of_features < 0)
		{
			parsing_done = true;
			return NULL;
		}



		if (example_type == E_LABELLED)
		{
			((LabelledExample*) current_example)->label = current_label;
			((LabelledExample*) current_example)->feature_vector = current_feature_vector;
			((LabelledExample*) current_example)->dimensions = current_number_of_features;
		}

		else
		{
			((UnlabelledExample*) current_example)->feature_vector = current_feature_vector;
			((UnlabelledExample*) current_example)->dimensions = current_number_of_features;
		}


		// Now copy the example into the buffer
		copy_example_into_buffer(current_example);

		buffer_increment_write_index();
		number_of_vectors_parsed++;

	}

	return NULL;

}

void* CInputParser::get_next_example()
{
	// Return the next unused example from the buffer

	void *example;

	if (buffer_read_index < 0)
	{
		if (number_of_vectors_parsed > 0)
			buffer_increment_read_index();
		else
			return NULL;
	}


	if (parsing_done)
	{
		if (number_of_vectors_read == number_of_vectors_parsed)
		{
			reading_done = true;
		}
	}


	if (number_of_vectors_read == number_of_vectors_parsed)
	{
		return NULL;
	}


	if (is_example_used[buffer_read_index] == E_NOT_USED)
	{
		pthread_mutex_lock(&example_in_use_mutex[buffer_read_index]);

		example = ((char *) examples_buff + example_memsize * buffer_read_index);

		pthread_mutex_unlock(&example_in_use_mutex[buffer_read_index]);
		number_of_vectors_read++;
		return example;
	}

	else
	{
		return NULL;
	}

}

int32_t CInputParser::get_next_example_labelled(float64_t* &feature_vector, int32_t &length, float64_t &label)
{
	/* if reading is done, no more examples can be fetched. return 0
	   else, if example can be read, get the example and return 1.
	   otherwise, wait for further parsing, get the example and
	   return 1 */

	LabelledExample *example;

	while (1)
	{
		if (reading_done)
			return 0;

		example = (LabelledExample *) get_next_example();

		if (example == NULL)
			continue;
		else
			break;
	}


	feature_vector = example->feature_vector;
	length = example->dimensions;
	label = example->label;

	return 1;
}

int32_t CInputParser::get_next_example_unlabelled(float64_t* &feature_vector, int32_t &length)
{
	UnlabelledExample *example;

	while (1)
	{
		if (reading_done)
			return 0;

		example = (UnlabelledExample *) get_next_example();

		if (example == NULL)
			continue;
		else
			break;
	}


	feature_vector = example->feature_vector;
	length = example->dimensions;

	return 1;
}


void CInputParser::set_buffer_size(int32_t size)
{
	buffer_size = size;
}

void CInputParser::finalize_example()
{
  	pthread_mutex_lock(&example_in_use_mutex[buffer_read_index]);
	is_example_used[buffer_read_index] = E_USED;
	pthread_cond_signal(&example_in_use_condition[buffer_read_index]);
	pthread_mutex_unlock(&example_in_use_mutex[buffer_read_index]);
	buffer_increment_read_index();
}

void CInputParser::end_parser()
{
	pthread_join(parse_thread, NULL);
}


