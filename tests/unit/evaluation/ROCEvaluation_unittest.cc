/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/labels/BinaryLabels.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ROCEvaluation,one)
{
	index_t num_labels=10;
	CBinaryLabels* gt=new CBinaryLabels(num_labels);
	CROCEvaluation* roc=new CROCEvaluation();

	for (index_t i=0; i<num_labels; i++)
	{
		float64_t l=i%2==0 ? -1 : 1;
		gt->set_value(l, i);
		gt->set_label(i, l);
	}

	float64_t auc=roc->evaluate(gt, gt);
	EXPECT_EQ(auc, 1);

	SG_UNREF(roc);
	SG_UNREF(gt);
}
