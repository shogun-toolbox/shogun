/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Heiko Strathmann
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 *
 * This file demonstrates how a regular CDenseFeatures object can
 * be used as input for the StreamingFeatures framework, effectively
 * making it suitable for using online learning algorithms.
 */

#include <features/streaming/StreamingDenseFeatures.h>
#include <io/streaming/StreamingFileFromDenseFeatures.h>

#include <mathematics/Math.h>
#include <lib/common.h>
#include <io/SGIO.h>
#include <base/init.h>

#include <stdlib.h>
#include <stdio.h>

using namespace shogun;

#define NUM 10
#define DIMS 2
#define DIST 0.5

void gen_rand_data(SGMatrix<float64_t> feat, SGVector<float64_t> lab)
{
	for (int32_t i=0; i<NUM; i++)
	{
		if (i<NUM/2)
		{
			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0, 1.0)+DIST;

			if (lab.vector)
				lab[i]=0;
		}
		else
		{
			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=CMath::random(0.0, 1.0)-DIST;

			if (lab.vector)
				lab[i]=1;
		}
	}
	feat.display_matrix("feat");
	lab.display_vector("lab");
}

void test_general()
{
	SGMatrix<float64_t> feat(DIMS, NUM);
	SGVector<float64_t> lab(NUM);

	// Generate random data, features and labels
	gen_rand_data(feat, lab);

	// Create features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	SG_REF(features);
	features->set_feature_matrix(feat);

	// Create a StreamingDenseFeatures object which uses the above as input;
	// labels (float64_t*) are optional
	CStreamingDenseFeatures<float64_t>* streaming=new CStreamingDenseFeatures<
			float64_t>(features, lab);
	SG_REF(streaming);

	// Start parsing of the examples; in this case, it is trivial - returns each vector from the DenseFeatures object
	streaming->start_parser();

	int32_t counter=0;
	SG_SPRINT("Processing examples...\n\n");

	// Run a while loop over all the examples.  Note that since
	// features are "streaming", there is no predefined
	// number_of_vectors known to the StreamingFeatures object.
	// Thus, this loop must be used to iterate over all the
	// features
	while (streaming->get_next_example())
	{
		counter++;
		// Get the current vector; no other vector is accessible
		SGVector<float64_t> vec=streaming->get_vector();
		float64_t label=streaming->get_label();

		SG_SPRINT("Vector %d: [\t", counter);
		for (int32_t i=0; i<vec.vlen; i++)
		{
			SG_SPRINT("%f\t", vec.vector[i]);
		}
		SG_SPRINT("Label=%f\t", label);

		// Calculate dot product of the current vector (from
		// the StreamingFeatures object) with itself (the
		// vector passed as argument)
		float64_t dot_prod=streaming->dense_dot(vec.vector, vec.vlen);

		SG_SPRINT("]\nDot product of the vector with itself: %f", dot_prod);
		SG_SPRINT("\n\n");

		// Free the example, since we are done with processing it.
		streaming->release_example();
	}

	// Now that all examples are used, end the parser.
	streaming->end_parser();

	SG_UNREF(streaming);
	SG_UNREF(features);
}

void test_get_streamed_features()
{
	/* create streaming features from dense features and then make call and
	 * assert that data is equal */

	SGMatrix<float64_t> feat(DIMS, NUM);
	SGVector<float64_t> lab(NUM);

	// Generate random data, features and labels
	gen_rand_data(feat, lab);

	// Create features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	SG_REF(features);
	features->set_feature_matrix(feat);

	// Create a StreamingDenseFeatures object which uses the above as input;
	// labels (float64_t*) are optional
	CStreamingDenseFeatures<float64_t>* streaming=new CStreamingDenseFeatures<
			float64_t>(features, lab);
	SG_REF(streaming);

	streaming->start_parser();
	CDenseFeatures<float64_t>* dense=
			(CDenseFeatures<float64_t>*)streaming->get_streamed_features(NUM);

	streaming->end_parser();

	/* assert that matrices are equal */
	ASSERT(dense->get_feature_matrix().equals(feat));

	SG_UNREF(dense);



	SG_UNREF(features);
	SG_UNREF(streaming);
}

void test_get_streamed_features_too_many()
{
	/* create streaming features from dense features and then make call and
	 * assert that data is equal. requests more data than available */

	SGMatrix<float64_t> feat(DIMS, NUM);
	SGVector<float64_t> lab(NUM);

	// Generate random data, features and labels
	gen_rand_data(feat, lab);

	// Create features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>();
	SG_REF(features);
	features->set_feature_matrix(feat);

	// Create a StreamingDenseFeatures object which uses the above as input;
	// labels (float64_t*) are optional
	CStreamingDenseFeatures<float64_t>* streaming=new CStreamingDenseFeatures<
			float64_t>(features, lab);
	SG_REF(streaming);

	streaming->start_parser();

	/* request more features than available */
	CDenseFeatures<float64_t>* dense=
			(CDenseFeatures<float64_t>*)streaming->get_streamed_features(NUM+10);

	streaming->end_parser();

	/* assert that matrices are equal */
	ASSERT(dense->get_feature_matrix().equals(feat));

	SG_UNREF(dense);



	SG_UNREF(features);
	SG_UNREF(streaming);
}

int main()
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

	test_general();
	test_get_streamed_features();
	test_get_streamed_features_too_many();
//
	exit_shogun();
	return 0;
}

