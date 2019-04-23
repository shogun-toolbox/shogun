/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Abhijeet Kislay, Pan Deng, Sergey Lisitsyn
 */

#include <shogun/lib/config.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/preprocessor/FisherLDA.h>
#include <shogun/features/DenseFeatures.h>
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

	feat=DataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	for(int i=0; i<CLASSES; ++i)
		for(int j=0; j<NUM; ++j)
			lab[i*NUM+j]=double(i);

	// Create train labels
	MulticlassLabels* labels=new MulticlassLabels(lab);

	// Create train features
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(feat);

	// Initiate the FisherLDA class
	CFisherLDA* fisherlda=new CFisherLDA(AUTO_FLDA);
	fisherlda->fit(features, labels, 1);
	SGMatrix<float64_t> y = fisherlda->transform(features)
	                            ->as<DenseFeatures<float64_t>>()
	                            ->get_feature_matrix();

	// display output
	y.display_matrix();
}

int main(int argc, char ** argv)
{
	test();
	return 0;
}
