#include "features/Features.h"
#include "lib/io.h"
#include "features/StreamingFeatures.h"

#include <stdio.h>
#include <string.h>

using namespace shogun;

void CStreamingFeatures::init()
{
	// initialize stuff
	current_feature_vector = NULL;
	working_file = NULL;
	current_label = -1;
	current_length = -1;
	
}

CStreamingFeatures::CStreamingFeatures()
{
	init();
}

CStreamingFeatures::CStreamingFeatures(FILE* file, int32_t buffer_size)
{
	// For now, assume file input
	
	init();
	working_file = file;
	parser.init(file);
	parser.set_buffer_size(buffer_size);
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

// Behaviour of get_next_feature_vector:
// if parsing is complete,
// -> if reading is complete, return 0
// -> else, fetch the example, return 1
// if parsing is incomplete,
// -> if reading is incomplete, fetch example, return 1
// -> if reading is complete (upto the last parsed example),
// -----> wait for the parser to get more examples
// -----> fetch the example and return 1

int32_t CStreamingFeatures::get_next_feature_vector(float64_t* &feature_vector, int32_t &length, int32_t &label)
{
	int32_t ret_value;

	ret_value =	parser.get_next_example(feature_vector, length, label);

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
	// parser can now remove this example from the buffer.
	
	return ret_value;
}
