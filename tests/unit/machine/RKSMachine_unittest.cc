/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evangelos Anagnostopoulos
 */

#include <shogun/machine/RKSMachine.h>

#include <gtest/gtest.h>

using namespace shogun;

SGVector<float64_t> p()
{
	SGVector<float64_t> params(2);
	params[0] = 1;
	params[1] = 0;
	return params;
}

float64_t phi(SGVector<float64_t> x, SGVector<float64_t> w)
{
	if (x[(int32_t) w[0]] > w[1])
		return 1;
	return 0;	
}

TEST(RKSMachine, creating_features)
{
	SGMatrix<float64_t> mat(5, 10);
	SGVector<float64_t> labs(10);
	for (index_t v=0; v<10; v++)
	{
		for (index_t d=0; d<5; d++)
			mat(d,v) = 0;

		if (v%2==0)
		{
			mat(1,v) = 2;
			labs[v] = 1;
		}
		else
			labs[v] = -1;
	}

	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(mat);
	CBinaryLabels* labels = new CBinaryLabels(labs);
	CRKSMachine* machine = new CRKSMachine(feats, labels, 10, &phi, &p); 
	CDenseFeatures<float64_t>* comp_feats = machine->get_features();
	EXPECT_EQ(comp_feats->get_num_vectors(), 10);
	EXPECT_EQ(comp_feats->get_dim_feature_space(), 10);

	for (index_t i=0; i<comp_feats->get_num_vectors(); i++)
	{
		SGVector<float64_t> vec = comp_feats->get_feature_vector(i);
		for (index_t j=0; j<comp_feats->get_dim_feature_space(); j++)
		{
			if (i%2==0)
				EXPECT_EQ(vec[j], 1);
			else
				EXPECT_EQ(vec[j], 0);
		}
	}

	SG_UNREF(machine);
	SG_UNREF(comp_feats);
	SG_UNREF(feats);
}
