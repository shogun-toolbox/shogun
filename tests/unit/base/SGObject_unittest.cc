/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Heiko Strathmann
 */

#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGObject,equals_null)
{
	CBinaryLabels* labels=new CBinaryLabels(10);

	EXPECT_FALSE(labels->equals(NULL));

	SG_UNREF(labels);
}

TEST(SGObject,equals_different_name)
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	CRegressionLabels* labels2=new CRegressionLabels(10);

	EXPECT_FALSE(labels->equals(labels2));

	SG_UNREF(labels);
	SG_UNREF(labels2);
}


