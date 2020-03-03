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

StreamingFileFromFeatures::StreamingFileFromFeatures(const std::shared_ptr<Features>& feat)
	: StreamingFile(), features(feat)
{
	labels=NULL;
}

StreamingFileFromFeatures::StreamingFileFromFeatures(const std::shared_ptr<Features>& feat, float64_t* lab)
    : StreamingFileFromFeatures(feat)
{
	labels=lab;
}

StreamingFileFromFeatures::~StreamingFileFromFeatures()
{
	features=NULL;
	labels=NULL;
}
