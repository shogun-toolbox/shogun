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

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void check_transposed(CDenseFeatures<int32_t>* features)
{
	CDenseFeatures<int32_t>* transposed=features->get_transposed();
	CDenseFeatures<int32_t>* double_transposed=transposed->get_transposed();

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> orig_vec=features->get_feature_vector(i);
		SGVector<int32_t> new_vec=double_transposed->get_feature_vector(i);

		ASSERT(orig_vec.vlen==new_vec.vlen);

		for (index_t j=0; j<orig_vec.vlen; j++)
			ASSERT(orig_vec.vector[j]==new_vec.vector[j]);

		/* not necessary since feature matrix is in memory. for documentation */
		features->free_feature_vector(orig_vec,i);
		double_transposed->free_feature_vector(new_vec, i);
	}

	SG_UNREF(transposed);
	SG_UNREF(double_transposed);
}

const int32_t num_vectors=6;
const int32_t dim_features=6;

void test()
{
	const int32_t num_subset_idx=CMath::random(1, num_vectors);

	/* create feature data matrix */
	SGMatrix<int32_t> data(dim_features, num_vectors);

	/* fill matrix with random data */
	for (index_t i=0; i<num_vectors; ++i)
	{
		for (index_t j=0; j<dim_features; ++j)
			data.matrix[i*dim_features+j]=CMath::random(-5, 5);
	}

	/* create simple features */
	CDenseFeatures<int32_t>* features=new CDenseFeatures<int32_t> (data);
	SG_REF(features);

	/* print feature matrix */
	SGMatrix<int32_t>::display_matrix(data.matrix, data.num_rows, data.num_cols,
			"feature matrix");

	/* create subset indices */
	SGVector<index_t> subset_idx(SGVector<index_t>::randperm(num_subset_idx),
			num_subset_idx);

	/* print subset indices */
	SGVector<index_t>::display_vector(subset_idx.vector, subset_idx.vlen, "subset indices");

	/* apply subset to features */
	SG_SPRINT("\n\n-------------------\n"
			"applying subset to features\n"
			"-------------------\n");
	features->add_subset(subset_idx);

	/* do some stuff do check and output */
	ASSERT(features->get_num_vectors()==num_subset_idx);

	/* check get_Transposed method */
	SG_SPRINT("checking transpose...");
	check_transposed(features);
	SG_SPRINT("does work\n");

	SG_SPRINT("features->get_num_vectors(): %d\n", features->get_num_vectors());

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_feature_vector(i);
		SG_SPRINT("vector %d: ", i);
		SGVector<int32_t>::display_vector(vec.vector, vec.vlen);

		for (index_t j=0; j<dim_features; ++j)
			ASSERT(vec.vector[j]==data.matrix[subset_idx.vector[i]*num_vectors+j]);

		/* not necessary since feature matrix is in memory. for documentation */
		features->free_feature_vector(vec, i);
	}

	/* remove features subset */
	SG_SPRINT("\n\n-------------------\n"
			"removing subset from features\n"
			"-------------------\n");
	features->remove_all_subsets();

	/* do some stuff do check and output */
	ASSERT(features->get_num_vectors()==num_vectors);
	SG_SPRINT("features->get_num_vectors(): %d\n", features->get_num_vectors());

	/* check get_Transposed method */
	SG_SPRINT("checking transpose...");
	check_transposed(features);
	SG_SPRINT("does work\n");

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_feature_vector(i);
		SG_SPRINT("vector %d: ", i);
		SGVector<int32_t>::display_vector(vec.vector, vec.vlen);

		for (index_t j=0; j<dim_features; ++j)
			ASSERT(vec.vector[j]==data.matrix[i*num_vectors+j]);

		/* not necessary since feature matrix is in memory. for documentation */
		features->free_feature_vector(vec, i);
	}

	SG_UNREF(features);
}

int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();
	return 0;
}

