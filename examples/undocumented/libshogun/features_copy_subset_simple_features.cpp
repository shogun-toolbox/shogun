/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <base/init.h>
#include <features/DenseFeatures.h>
#include <features/Subset.h>

using namespace shogun;

void test()
{

	SGMatrix<float64_t> data(3, 10);
	CDenseFeatures<float64_t>* f=new CDenseFeatures<float64_t>(data);
	SGVector<float64_t>::range_fill_vector(data.matrix, data.num_cols*data.num_rows, 1.0);
	SGMatrix<float64_t>::display_matrix(data.matrix, data.num_rows, data.num_cols,
			"original feature data");

	index_t offset_subset=1;
	SGVector<index_t> feature_subset(8);
	SGVector<index_t>::range_fill_vector(feature_subset.vector, feature_subset.vlen,
			offset_subset);
	SGVector<index_t>::display_vector(feature_subset.vector, feature_subset.vlen,
			"feature subset");

	f->add_subset(feature_subset);
	SG_SPRINT("feature vectors after setting subset on original data:\n");
	for (index_t i=0; i<f->get_num_vectors(); ++i)
	{
		SGVector<float64_t> vec=f->get_feature_vector(i);
		SG_SPRINT("%i: ", i);
		SGVector<float64_t>::display_vector(vec.vector, vec.vlen);
		f->free_feature_vector(vec, i);
	}

	index_t offset_copy=2;
	SGVector<index_t> feature_copy_subset(4);
	SGVector<index_t>::range_fill_vector(feature_copy_subset.vector,
			feature_copy_subset.vlen, offset_copy);
	SGVector<index_t>::display_vector(feature_copy_subset.vector, feature_copy_subset.vlen,
			"indices that are to be copied");

	CDenseFeatures<float64_t>* subset_copy=
			(CDenseFeatures<float64_t>*)f->copy_subset(feature_copy_subset);

	SGMatrix<float64_t> subset_copy_matrix=subset_copy->get_feature_matrix();
	SGMatrix<float64_t>::display_matrix(subset_copy_matrix.matrix,
			subset_copy_matrix.num_rows, subset_copy_matrix.num_cols,
			"copy matrix");

	index_t num_its=subset_copy_matrix.num_rows*subset_copy_matrix.num_cols;
	for (index_t i=0; i<num_its; ++i)
	{
		index_t idx=i+(offset_copy+offset_subset)*subset_copy_matrix.num_rows;
		ASSERT(subset_copy_matrix.matrix[i]==data.matrix[idx]);
	}

	SG_UNREF(f);
	SG_UNREF(subset_copy);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();
	return 0;
}

