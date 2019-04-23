/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Sergey Lisitsyn
 */

#include <shogun/io/streaming/StreamingFileFromFeatures.h>

using namespace shogun;

StreamingFileFromFeatures::StreamingFileFromFeatures()
	: StreamingFile()
{
	features=NULL;
	labels=NULL;
}

StreamingFileFromFeatures::StreamingFileFromFeatures(std::shared_ptr<Features> feat)
	: StreamingFile()
{
	features=feat;
	labels=NULL;
}

StreamingFileFromFeatures::StreamingFileFromFeatures(std::shared_ptr<Features> feat, float64_t* lab)
	: StreamingFile()
{
	features=feat;
	labels=lab;
}

StreamingFileFromFeatures::~StreamingFileFromFeatures()
{
	features=NULL;
	labels=NULL;
}
