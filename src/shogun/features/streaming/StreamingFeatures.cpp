/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Viktor Gal
 */
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

StreamingFeatures::StreamingFeatures() : Features()
{
	working_file=NULL;
}

StreamingFeatures::~StreamingFeatures()
{
}

void StreamingFeatures::set_read_functions()
{
	set_vector_reader();
	set_vector_and_label_reader();
}

bool StreamingFeatures::get_has_labels()
{
	return has_labels;
}

bool StreamingFeatures::is_seekable()
{
	return seekable;
}

void StreamingFeatures::reset_stream()
{
	not_implemented(SOURCE_LOCATION);
	return;
}
