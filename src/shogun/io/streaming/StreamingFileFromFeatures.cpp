/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
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
