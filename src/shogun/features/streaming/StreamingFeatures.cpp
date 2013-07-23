/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Written (W) 2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
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
	SG_DEBUG("entering CStreamingFeatures::~CStreamingFeatures()\n")
	SG_UNREF(working_file);
	SG_DEBUG("leaving CStreamingFeatures::~CStreamingFeatures()\n")
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
	SG_NOTIMPLEMENTED
	return;
}
