/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */

#include <shogun/features/streaming/StreamingDenseFeatures.h>
#include <shogun/io/streaming/StreamingFileFromDenseFeatures.h>

#include <shogun/mathematics/Math.h>
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/ShogunEnv.h>

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
				feat[i*DIMS+j]=Math::random(0.0, 1.0)+DIST;

			if (lab.vector)
				lab[i]=0;
		}
		else
		{
			for (int32_t j=0; j<DIMS; j++)
				feat[i*DIMS+j]=Math::random(0.0, 1.0)-DIST;

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
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat);

	// Create a StreamingDenseFeatures object which uses the above as input;
	// labels (float64_t*) are optional
	auto streaming=std::make_shared<StreamingDenseFeatures<
			float64_t>>(features, lab);

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
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat);

	// Create a StreamingDenseFeatures object which uses the above as input;
	// labels (float64_t*) are optional
	auto streaming=std::make_shared<StreamingDenseFeatures<
			float64_t>>(features, lab);

	streaming->start_parser();
	auto dense=
		streaming->get_streamed_features(NUM)->as<DenseFeatures<float64_t>>();

	streaming->end_parser();

	/* assert that matrices are equal */
	ASSERT(dense->get_feature_matrix().equals(feat));




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
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat);

	// Create a StreamingDenseFeatures object which uses the above as input;
	// labels (float64_t*) are optional
	auto streaming=std::make_shared<StreamingDenseFeatures<
			float64_t>>(features, lab);

	streaming->start_parser();

	/* request more features than available */
	auto dense=
			streaming->get_streamed_features(NUM+10)->as<DenseFeatures<float64_t>>();

	streaming->end_parser();

	/* assert that matrices are equal */
	ASSERT(dense->get_feature_matrix().equals(feat));




}

int main()
{
	env()->io()->set_loglevel(MSG_DEBUG);

	test_general();
	test_get_streamed_features();
	test_get_streamed_features_too_many();
//
	return 0;
}

