/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012-2013 Heiko Strathmann
 */

#include <shogun/labels/BinaryLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(BinaryLabels,scores_to_probabilities)
{
	CBinaryLabels* labels=new CBinaryLabels(10);
	labels->set_values(SGVector<float64_t>(labels->get_num_labels()));

	for (index_t i=0; i<labels->get_num_labels(); ++i)
		labels->set_value(i%2==0 ? 1 : -1, i);

	//labels->get_values().display_vector("scores");
	// call with 0,0 to make the method compute sigmoid parameters itself
	// g-test somehow does not allow std parameters
	labels->scores_to_probabilities(0,0);

	/* only two probabilities will be the result. Results from implementation that
	 * comes with the original paper, see BinaryLabels documentation */
	EXPECT_NEAR(labels->get_value(0), 0.8571428439385661, 10E-15);
	EXPECT_NEAR(labels->get_value(1), 0.14285715606143384, 10E-15);

	SG_UNREF(labels);
}
