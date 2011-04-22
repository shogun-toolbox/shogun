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

CStreamingFeatures::CStreamingFeatures(CStreamingFile* file, bool is_labelled = true)
{
	init();
	has_labels = is_labelled;
	working_file = file;
	parser.init(file, is_labelled);
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

int32_t CStreamingFeatures::get_next_feature_vector(float64_t* &feature_vector, int32_t &length, float64_t &label)
{
	int32_t ret_value;

	ret_value = parser.get_next_example_labelled(feature_vector, length, label);

	// If all examples have been fetched, return 0.
	if (ret_value == 0)
		return 0;

	// Now set current_{feature_vector, label, length} for the object
	current_length = length;
	current_label = label;

	// in case current_feature_vector isn't initialized
	if (current_feature_vector == NULL)
		current_feature_vector = new float64_t[length];

	memcpy(current_feature_vector, feature_vector, length*sizeof(float64_t));

	feature_vector = current_feature_vector; // effectively, this address is constant

	parser.finalize_example();

	return ret_value;
}

int32_t CStreamingFeatures::get_next_feature_vector(float64_t* &feature_vector, int32_t &length)
{
	int32_t ret_value;

	ret_value = parser.get_next_example_unlabelled(feature_vector, length);

	// If all examples have been fetched, return 0.
	if (ret_value == 0)
		return 0;

	// Now set current_{feature_vector, length} for the object
	current_length = length;

	// in case current_feature_vector isn't initialized
	if (current_feature_vector == NULL)
		current_feature_vector = new float64_t[length];

	memcpy(current_feature_vector, feature_vector, length*sizeof(float64_t));

	feature_vector = current_feature_vector; // effectively, this address is constant

	parser.finalize_example();

	return ret_value;
}
