/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Fernando José Iglesias García
 * Copyright (C) 2012 Fernando José Iglesias García
 */

#include <shogun/base/init.h>
#include <shogun/classifier/QDA.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

#define	NUM  100
#define DIMS 2
#define DIST 0.5

void gen_rand_data(SGVector< float64_t > lab, SGMatrix< float64_t > feat)
{
	for (int32_t i = 0; i < NUM; i++)
	{
		if (i < NUM/2)
		{
			lab[i] = 0.0;

			for (int32_t j = 0; j < DIMS; j++)
				feat[i*DIMS + j] = CMath::random(0.0,1.0) + DIST;
		}
		else
		{
			lab[i] = 1.0;

			for (int32_t j = 0; j < DIMS; j++)
				feat[i*DIMS + j] = CMath::random(0.0,1.0) - DIST;
		}
	}
}

int main(int argc, char ** argv)
{
	const int32_t feature_cache = 0;

	init_shogun_with_defaults();

	SGVector< float64_t > lab(NUM);
	SGMatrix< float64_t > feat(NUM, DIMS);

	gen_rand_data(lab, feat);

	// Create train labels
	CLabels* labels = new CLabels(lab);

	// Create train features
	CSimpleFeatures< float64_t >* features = new CSimpleFeatures< float64_t >(feature_cache);
	features->set_feature_matrix(feat.matrix, DIMS, NUM);

	// Create QDA classifier
	CQDA* qda = new CQDA(features, labels);
	SG_REF(qda);
	qda->train();

	// Classify and display output
	CLabels* out_labels = qda->apply();
	SG_REF(out_labels);

	// Free memory
	SG_UNREF(out_labels);
	SG_UNREF(qda);

	exit_shogun();

	return 0;
}
