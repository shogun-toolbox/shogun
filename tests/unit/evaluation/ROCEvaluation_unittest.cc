/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 */

#include <shogun/base/init.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/evaluation/ROCEvaluation.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ROCEvaluation,one)
{
	int num_labels = 10;
	CBinaryLabels* gt = new CBinaryLabels(num_labels);
	CROCEvaluation* roc = new CROCEvaluation();

	for (int i = 0; i < num_labels; i++) {
		int l = (CMath::random(-1.0, 1.0) < 0 ? -1 : 1);
		gt->set_label(i, l);
	}

	roc->evaluate(gt, gt);

	EXPECT_EQ(roc->get_auROC(), 1);
}