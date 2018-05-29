/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Abhijeet Kislay, Pan Deng, Sergey Lisitsyn
 */

#include <shogun/base/init.h>

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

	feat=CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);
	for(int i=0; i<CLASSES; ++i)
		for(int j=0; j<NUM; ++j)
			lab[i*NUM+j]=double(i);

	// Create train labels
	CMulticlassLabels* labels=new CMulticlassLabels(lab);
	SG_REF(labels)

	// Create train features
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(feat);
	SG_REF(features)

	// Initiate the FisherLDA class
	CFisherLDA* fisherlda=new CFisherLDA(AUTO_FLDA);
	SG_REF(fisherlda)
	fisherlda->fit(features, labels, 1);
	SGMatrix<float64_t> y = fisherlda->transform(features)
	                            ->as<CDenseFeatures<float64_t>>()
	                            ->get_feature_matrix();

	// display output
	y.display_matrix();
	SG_UNREF(fisherlda)
	SG_UNREF(features)
	SG_UNREF(labels)
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();
	test();
	exit_shogun();
	return 0;
}
