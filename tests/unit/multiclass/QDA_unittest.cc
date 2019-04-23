/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann
 */
#include <gtest/gtest.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/multiclass/QDA.h>

#define NUM  50
#define DIMS 2
#define CLASSES 2

using namespace shogun;

#ifdef HAVE_LAPACK
TEST(QDA, train_and_apply)
{
	std::mt19937_64 prng(240);

	SGVector< float64_t > lab(CLASSES*NUM);
	SGMatrix< float64_t > feat(DIMS, CLASSES*NUM);

	feat = DataGenerator::generate_gaussians(NUM,CLASSES,DIMS,prng);
	for( int i = 0 ; i < CLASSES ; ++i )
		for( int j = 0 ; j < NUM ; ++j )
			lab[i*NUM+j] = double(i);

	auto labels = std::make_shared<MulticlassLabels>(lab);
	auto features = std::make_shared<DenseFeatures< float64_t >>(feat);

	auto qda = std::make_shared<QDA>(features, labels);

	qda->train();

	auto output = qda->apply()->as<MulticlassLabels>();
	// Test
	for ( index_t i = 0; i < CLASSES*NUM; ++i )
		EXPECT_EQ(output->get_label(i), labels->get_label(i));

}
#endif // HAVE_LAPACK
