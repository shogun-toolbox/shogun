/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * This file demonstrates how a regular CSimpleFeatures object can
 * be used as input for the StreamingFeatures framework, effectively
 * making it suitable for using online learning algorithms.
 */

#include <shogun/features/StreamingSimpleFeatures.h>
#include <shogun/lib/StreamingFileFromSimpleFeatures.h>

#include <shogun/lib/Mathematics.h>
#include <shogun/lib/common.h>
#include <shogun/base/init.h>

#include <stdlib.h>
#include <stdio.h>

using namespace shogun;

#define NUM 100
#define DIMS 2
#define DIST 0.5

float64_t* feat;

void gen_rand_data()
{
	feat=new float64_t[NUM*DIMS];

	for (int32_t i=0; i<NUM; i++)
	{
		if (i<NUM/2)
		{
			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0,1.0)+DIST;
		}
		else
		{
			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0,1.0)-DIST;
		}
	}
	CMath::display_matrix(feat,DIMS, NUM);
}

int main()
{
	init_shogun();

	// Generate random data
	gen_rand_data();

	// Create features
	CSimpleFeatures<float64_t>* features = new CSimpleFeatures<float64_t>();
	SG_REF(features);
	features->set_feature_matrix(feat, DIMS, NUM);

	// Make a file-like object for streaming vector-by-vector from the CSimpleFeatures object
	CStreamingFileFromSimpleFeatures* simplefeatures_input = new CStreamingFileFromSimpleFeatures(features);
	SG_REF(simplefeatures_input);

	// Create a StreamingSimpleFeatures object which uses the above as input
	CStreamingSimpleFeatures<float64_t>* streaming_simple = new CStreamingSimpleFeatures<float64_t>(simplefeatures_input);
	SG_REF(streaming_simple);

	// Start parsing of the examples; in this case, it is trivial - returns each vector from the SimpleFeatures object
	streaming_simple->start_parser();

	int32_t counter=0;
	SG_PRINT("Processing examples...\n\n");
	
	// Run a while loop over all the examples.  Note that since
	// features are "streaming", there is no predefined
	// number_of_vectors known to the StreamingFeatures object.
	// Thus, this loop must be used to iterate over all the
	// features.
	while (streaming_simple->get_next_example())
	{
		counter++;
		// Get the current vector; no other vector is accessible
		SGVector<float64_t> vec = streaming_simple->get_vector();
		
		SG_PRINT("Vector %d: [\t", counter);
		for (int32_t i=0; i<vec.vlen; i++)
		{
			SG_PRINT("%f\t", vec.vector[i]);
		}
		
		// Calculate dot product of the current vector (from
		// the StreamingFeatures object) with itself (the
		// vector passed as argument)
		float64_t dot_prod = streaming_simple->dense_dot(vec);

		SG_PRINT("]\nDot product of the vector with itself: %f", dot_prod);
		SG_PRINT("\n\n");

		// Free the example, since we are done with processing it.
		streaming_simple->release_example();
	}

	// Now that all examples are used, end the parser.
	streaming_simple->end_parser();

	SG_UNREF(streaming_simple);
	SG_UNREF(simplefeatures_input);
	SG_UNREF(features);
	
	exit_shogun();
	return 0;
}
