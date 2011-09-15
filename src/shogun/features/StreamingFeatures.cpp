#include <shogun/features/StreamingFeatures.h>

using namespace shogun;

CStreamingFeatures::CStreamingFeatures() : CFeatures()
{
}

CStreamingFeatures::CStreamingFeatures(CStreamingFile* file,
		bool is_labelled, int32_t size) : CFeatures()
{
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
	SG_NOTIMPLEMENTED;
	return;
}
