/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Sergey Lisitsyn
 */

#include <shogun/io/streaming/StreamingFileFromFeatures.h>

using namespace shogun;

CStreamingFileFromFeatures::CStreamingFileFromFeatures()
	: CStreamingFile()
{
	features=NULL;
	labels=NULL;
}

CStreamingFileFromFeatures::CStreamingFileFromFeatures(CFeatures* feat)
	: CStreamingFile()
{
	features=feat;
	labels=NULL;
}

CStreamingFileFromFeatures::CStreamingFileFromFeatures(CFeatures* feat, float64_t* lab)
	: CStreamingFile()
{
	features=feat;
	labels=lab;
}

CStreamingFileFromFeatures::~CStreamingFileFromFeatures()
{
	features=NULL;
	labels=NULL;
}
