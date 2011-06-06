/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "features/Features.h"
#include "lib/io.h"
#include "features/StreamingFeatures.h"

#include <stdio.h>
#include <string.h>

using namespace shogun;

void CStreamingFeatures::init()
{
	current_feature_vector = NULL;
	working_file = NULL;
	current_label = -1;
	current_length = -1;
}

CStreamingFeatures::CStreamingFeatures()
{
	init();
}

CStreamingFeatures::CStreamingFeatures(CStreamingFile* file, bool is_labelled, int32_t size)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled, size);
}

CStreamingFeatures::~CStreamingFeatures()
{
	parser.end_parser();
}

void CStreamingFeatures::start_parser()
{
	// start parser in another thread
	if (!parser.is_running())
		parser.start_parser();
}

void CStreamingFeatures::end_parser()
{
	parser.end_parser();
}

int32_t CStreamingFeatures::fetch_example()
{
  int32_t ret_value;

  ret_value = parser.get_next_example(current_feature_vector, current_length, current_label);

  if (ret_value == 0)
    return 0;

  return ret_value;
}

SGVector<float64_t> CStreamingFeatures::get_vector()
{
  SGVector<float64_t> vec;
  vec.vector=current_feature_vector;
  vec.length=current_length;

  return vec;
}

float64_t CStreamingFeatures::get_label()
{
  return current_label;
}

/*int32_t CStreamingFeatures::get_next_feature_vector(float64_t* &feature_vector, int32_t &length, float64_t &label)
{
	int32_t ret_value;

	ret_value = parser.get_next_example(feature_vector, length, label);

	// If all examples have been fetched, return 0.
	if (ret_value == 0)
		return 0;

	// Now set current_{feature_vector, label, length} for the object
	current_length = length;
	current_label = label;
	current_feature_vector = feature_vector;
	
	printf("Setting values: length = %d, label = %f, fv = %f", length, label, feature_vector[0]);
	return ret_value;
}

int32_t CStreamingFeatures::get_next_feature_vector(float64_t* &feature_vector, int32_t &length)
{
	int32_t ret_value;

	ret_value = parser.get_next_example(feature_vector, length);

	// If all examples have been fetched, return 0.
	if (ret_value == 0)
		return 0;

	// Now set current_{feature_vector, length} for the object
	current_length = length;
	current_feature_vector = feature_vector;

	return ret_value;
	}*/

void CStreamingFeatures::release_example()
{
	parser.finalize_example();
}
