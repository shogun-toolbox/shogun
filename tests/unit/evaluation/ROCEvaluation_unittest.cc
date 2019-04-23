/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Heiko Strathmann, Viktor Gal
 */

#include <shogun/labels/BinaryLabels.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ROCEvaluation,one)
{
	index_t num_labels=10;
	auto gt=std::make_shared<BinaryLabels>(num_labels);
	auto roc=std::make_shared<ROCEvaluation>();

	for (index_t i=0; i<num_labels; i++)
	{
		float64_t l=i%2==0 ? -1 : 1;
		gt->set_value(l, i);
		gt->set_label(i, l);
	}

	float64_t auc=roc->evaluate(gt, gt);
	EXPECT_EQ(auc, 1);

	
	
}
