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
}

CStreamingFeatures::~CStreamingFeatures()
{
	SG_UNREF(working_file);
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
CStreamingFeatures* CStreamingFeatures::from_non_streaming(
			CFeatures* source_features)
{
	CStreamingFeatures* result=NULL;

	REQUIRE(source_features, "CStreamingFeatures::from_non_streaming(): "
			"features required!\n");

	/* please help here! This must be solved better. Heiko Strathmann */
	if (source_features->get_feature_type()==F_DREAL)
	{
		CDenseFeatures<float64_t>* dense_features=
				dynamic_cast<CDenseFeatures<float64_t>*>(source_features);

		REQUIRE(dense_features, "CStreamingFeatures::from_non_streaming(): "
				"Provided features \"%s\" not supported!\n",
				source_features->get_name());

		result=new CStreamingDenseFeatures<float64_t>(dense_features);
	}
	else
	{
		SG_SERROR("CStreamingFeaturess::from_non_streaming(): Currently, only "
				"float64_t dense features suppoted!\n");
	}

	return result;
}
