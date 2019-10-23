/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Viktor Gal
 */
#include <shogun/features/streaming/StreamingFeatures.h>
#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

CStreamingFeatures::CStreamingFeatures() : CFeatures()
{
	working_file=NULL;
}

CStreamingFeatures::~CStreamingFeatures()
{
	SG_TRACE("entering CStreamingFeatures::~CStreamingFeatures()");
	SG_UNREF(working_file);
	SG_TRACE("leaving CStreamingFeatures::~CStreamingFeatures()");
}

void CStreamingFeatures::set_read_functions()
{
	set_vector_reader();
	set_vector_and_label_reader();
}

bool CStreamingFeatures::get_has_labels()
{
	return has_labels;
}

bool CStreamingFeatures::is_seekable()
{
	return seekable;
}

void CStreamingFeatures::reset_stream()
{
	not_implemented(SOURCE_LOCATION);
	return;
}
