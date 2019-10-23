/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Sergey Lisitsyn
 */

#include <shogun/io/streaming/StreamingFileFromFeatures.h>

#include <utility>

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
	features=std::move(feat);
	labels=NULL;
}

StreamingFileFromFeatures::StreamingFileFromFeatures(std::shared_ptr<Features> feat, float64_t* lab)
	: StreamingFile()
{
	features=std::move(feat);
	labels=lab;
}

StreamingFileFromFeatures::~StreamingFileFromFeatures()
{
	features=NULL;
	labels=NULL;
}
