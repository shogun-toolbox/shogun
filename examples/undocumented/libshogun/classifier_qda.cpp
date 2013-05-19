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
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/QDA.h>
#include <shogun/features/DenseFeatures.h>
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

void test()
{
	SGVector< float64_t > lab(NUM);
	SGMatrix< float64_t > feat(DIMS, NUM);

	gen_rand_data(lab, feat);

	// Create train labels
	CMulticlassLabels* labels = new CMulticlassLabels(lab);

	// Create train features
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);

	// Create QDA classifier
	CQDA* qda = new CQDA(features, labels);
	SG_REF(qda);
	qda->train();

	// Classify and display output
	CDenseFeatures< float64_t >* test_features = new CDenseFeatures< float64_t >(feat);
	CMulticlassLabels* out_labels = CLabelsFactory::to_multiclass(lda->apply(test_features));
	SG_REF(out_labels);

	// Free memory
	SG_UNREF(out_labels);
	SG_UNREF(qda);
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}
