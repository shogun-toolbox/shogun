/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Sergey Lisitsyn
 */

#include <shogun/base/init.h>
#include <shogun/labels/MulticlassLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MulticlassLabelsTest,confidences)
{
	const int n_labels = 3;
	const int n_classes = 4;

	CMulticlassLabels* labels = new CMulticlassLabels(n_labels);

	EXPECT_NO_THROW(labels->allocate_confidences_for(n_classes));

	for (int i=0; i<n_labels; i++)
		EXPECT_EQ(labels->get_multiclass_confidences(i).size(),n_classes);

	for (int i=0; i<n_labels; i++)
	{
		SGVector<float64_t> confs(n_classes);
		confs.zero();
		confs[i%n_classes] = 1.0;

		labels->set_multiclass_confidences(i,confs);

		SGVector<float64_t> obtained_confs = labels->get_multiclass_confidences(i);
		for (int j=0; j<n_classes; j++)
		{
			if (j==i%n_classes)
				EXPECT_NEAR(obtained_confs[j],1.0,1e-9);
			else
				EXPECT_NEAR(obtained_confs[j],0.0,1e-9);
		}
	}
	SG_UNREF(labels);
}
