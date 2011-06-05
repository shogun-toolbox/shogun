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
	init(NULL, true, PARSER_DEFAULT_BUFFSIZE);
}

CInputParser::~CInputParser()
{
	end_parser();
	
	delete current_example;
	delete examples_buff;
}

void CInputParser::init(CStreamingFile* input_file, bool is_labelled = true, int32_t size = PARSER_DEFAULT_BUFFSIZE)
{
	input_source = input_file;

	if (is_labelled == true)
		example_type = E_LABELLED;
	else
		example_type = E_UNLABELLED;

	examples_buff = new ParseBuffer(size);
	current_example = new example();
	
	parsing_done = false;
	reading_done = false;
	number_of_vectors_parsed = 0;
	number_of_vectors_read = 0;

	current_len = -1;
	current_label = -1;
	current_feature_vector = NULL;
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


int32_t CInputParser::get_vector_and_label(float64_t* &feature_vector,
										   int32_t &length,
										   float64_t &label)
{
	input_source->get_real_vector(feature_vector, length);
	/* The get_real_vector call should be replaced with
	   a dynamic call depending on the type of feature. */

	if (length < 2)
	{
		// Problem reading the example
		parsing_done=true;
		return 0;
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
		// Problem reading the example
		parsing_done=true;
		return 0;
	}

	return 1;
}

void CInputParser::copy_example_into_buffer(example* ex)
{
	examples_buff->copy_example(ex);
}

void* CInputParser::main_parse_loop(void* params)
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

example* CInputParser::retrieve_example()
{
	// Return the next unused example from the buffer

	example *ex;
	
	if (number_of_vectors_parsed <= 0)
		return NULL;

	if (parsing_done)
	{
		if (number_of_vectors_read == number_of_vectors_parsed)
			reading_done = true;
	}

	if (number_of_vectors_read == number_of_vectors_parsed)
		return NULL;
	
	ex = examples_buff->fetch_example();
	number_of_vectors_read++;

	return ex;
}

int32_t CInputParser::get_next_example(float64_t* &fv, int32_t &length, float64_t &label)
{
	/* if reading is done, no more examples can be fetched. return 0
	   else, if example can be read, get the example and return 1.
	   otherwise, wait for further parsing, get the example and
	   return 1 */
	
	example *ex;

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

int32_t CInputParser::get_next_example(float64_t* &fv, int32_t &length)
{
	float64_t label_dummy;
	
	return get_next_example(fv, length, label_dummy);
}

void CInputParser::finalize_example()
{
	examples_buff->finalize_example();
}

void CInputParser::end_parser()
{
	pthread_join(parse_thread, NULL);
}
