/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>

using namespace shogun;

void test_dense_features()
{
	/* create two matrices, feature objects for them, call create_merged_copy,
	 * and check if it worked */

	index_t n_1=3;
	index_t n_2=4;
	index_t dim=2;

	SGMatrix<float64_t> data_1(dim,n_1);
	for (index_t i=0; i<dim*n_1; ++i)
		data_1.matrix[i]=i;

	data_1.display_matrix("data_1");

	SGMatrix<float64_t> data_2(dim,n_2);
	for (index_t i=0; i<dim*n_2; ++i)
		data_2.matrix[i]=CMath::randn_double();

	data_1.display_matrix("data_2");

	CDenseFeatures<float64_t>* features_1=new CDenseFeatures<float64_t>(data_1);
	CDenseFeatures<float64_t>* features_2=new CDenseFeatures<float64_t>(data_2);


	CFeatures* concatenation=features_1->create_merged_copy(features_2);

	SGMatrix<float64_t> concat_data=
			((CDenseFeatures<float64_t>*)concatenation)->get_feature_matrix();
	concat_data.display_matrix("concat_data");

	/* check for equality with data_1 */
	for (index_t i=0; i<dim*n_1; ++i)
		ASSERT(data_1.matrix[i]==concat_data.matrix[i]);

	/* check for equality with data_2 */
	for (index_t i=0; i<dim*n_2; ++i)
		ASSERT(data_2.matrix[i]==concat_data.matrix[n_1*dim+i]);

	SG_UNREF(concatenation);
	SG_UNREF(features_1);
	SG_UNREF(features_2);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	sg_io->set_loglevel(MSG_DEBUG);

	test_dense_features();

	exit_shogun();

	return 0;
}

