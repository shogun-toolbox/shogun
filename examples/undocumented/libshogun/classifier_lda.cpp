/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Kevin Hughes, Heiko Strathmann, Evgeniy Andreev, Soeren Sonnenburg, 
 *          Pan Deng
 */

#include <shogun/base/init.h>

#include <shogun/lib/config.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/MCLDA.h>
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
#ifdef HAVE_LAPACK
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
	auto output = multiclass_labels(lda->apply());
	SG_REF(output);
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels());

	// Free memory
	SG_UNREF(lda);
#endif
}

int main(int argc, char ** argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

