/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Kevin Hughes
 * Copyright (C) 2013 Kevin Hughes
 *
 * Thanks to Fernando Jose Iglesias Garcia (shogun)
 *           and Matthieu Perrot (scikit-learn)
 */

#include <base/init.h>

#include <lib/config.h>
#include <labels/MulticlassLabels.h>
#include <multiclass/MCLDA.h>
#include <features/DenseFeatures.h>
#include <io/SGIO.h>
#include <lib/common.h>
#include <features/DataGenerator.h>

using namespace shogun;

#define NUM  50
#define DIMS 2
#define CLASSES 2

void test()
{
#ifdef HAVE_LAPACK
#ifdef HAVE_EIGEN3
	SGVector< float64_t > lab(CLASSES*NUM);
	SGMatrix< float64_t > feat(DIMS, CLASSES*NUM);

	feat = CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	for( int i = 0 ; i < CLASSES ; ++i )
		for( int j = 0 ; j < NUM ; ++j )
			lab[i*NUM+j] = double(i);

	// Create train labels
	CMulticlassLabels* labels = new CMulticlassLabels(lab);

	// Create train features
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);

	// Create QDA classifier
	CMCLDA* lda = new CMCLDA(features, labels);
	SG_REF(lda);
	lda->train();

	// Classify and display output
	CMulticlassLabels* output=CLabelsFactory::to_multiclass(lda->apply());
	SG_REF(output);
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels());

	// Free memory
	SG_UNREF(output);
	SG_UNREF(lda);
#endif
#endif
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

