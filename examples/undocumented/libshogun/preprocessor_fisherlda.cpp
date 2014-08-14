/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Abhijeet Kislay
 * Copyright (C) 2014 Abhijeet Kislay
*/

#include <shogun/base/init.h>

#ifdef HAVE_EIGEN3
#include <shogun/lib/config.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/preprocessor/FisherLDA.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/common.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

#define NUM  50
#define DIMS 2
#define CLASSES 2

void test()
{
	SGVector<float64_t> lab(CLASSES*NUM);
	SGMatrix<float64_t> feat(DIMS, CLASSES*NUM);

	feat=CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	for(int i=0; i<CLASSES; ++i)
		for(int j=0; j<NUM; ++j)
			lab[i*NUM+j] = double(i);

	// Create train labels
	CMulticlassLabels* labels=new CMulticlassLabels(lab);

	// Create train features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat);

	CFisherLDA fisherlda(AUTO_FLDA);
	fisherlda.init(features, labels, 1);
	SGMatrix<float64_t> y=fisherlda.apply_to_feature_matrix(features);

	// display output
	y.display_matrix();
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();
	test();
	exit_shogun();
	return 0;
}
#endif
