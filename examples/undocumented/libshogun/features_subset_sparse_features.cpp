/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Heiko Strathmann
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include <shogun/base/init.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/Subset.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

const int32_t num_vectors=6;
const int32_t dim_features=6;

void check_transposed(CSparseFeatures<int32_t>* features)
{
	CSparseFeatures<int32_t>* transposed=features->get_transposed();
	CSparseFeatures<int32_t>* double_transposed=transposed->get_transposed();

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		int32_t len;
		bool free_1, free_2;
		SGSparseVectorEntry<int32_t>* orig_vec=
				features->get_sparse_feature_vector(i, len, free_1);
		SGSparseVectorEntry<int32_t>* new_vec=
				double_transposed->get_sparse_feature_vector(i, len, free_2);

		for (index_t j=0; j<len; j++)
			ASSERT(orig_vec[j].entry==new_vec[j].entry);

		/* not necessary since feature matrix is in memory. for documentation */
		features->free_sparse_feature_vector(orig_vec, i, free_1);
		double_transposed->free_sparse_feature_vector(new_vec, i, free_2);
	}

	SG_UNREF(transposed);
	SG_UNREF(double_transposed);
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);
	const int32_t num_subset_idx=CMath::random(1, num_vectors);

	/* create feature data matrix */
	SGMatrix<int32_t> data(dim_features, num_vectors);

	/* fill matrix with random data */
	for (index_t i=0; i<num_vectors*dim_features; ++i)
		data.matrix[i]=CMath::random(1, 9);

	/* create sparse features */
	CSparseFeatures<int32_t>* features=new CSparseFeatures<int32_t>(data);

	/* print dense feature matrix */
	CMath::display_matrix(data.matrix, data.num_rows, data.num_cols,
			"dense feature matrix");

	/* create subset indices */
	SGVector<index_t> subset_idx(CMath::randperm(num_subset_idx),
			num_subset_idx);

	/* print subset indices */
	CMath::display_vector(subset_idx.vector, subset_idx.vlen, "subset indices");

	/* apply subset to features */
	SG_SPRINT("\n-------------------\n"
	"applying subset to features\n"
	"-------------------\n");
	features->set_subset(new CSubset(subset_idx));

	/* do some stuff do check and output */
	ASSERT(features->get_num_vectors()==num_subset_idx);
	SG_SPRINT("features->get_num_vectors(): %d\n", features->get_num_vectors());

	/* check get_Transposed method */
	SG_SPRINT("checking transpose...");
	check_transposed(features);
	SG_SPRINT("does work\n");

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		int32_t len;
		bool free;
		SGSparseVectorEntry<int32_t>* vec=features->get_sparse_feature_vector(i,
				len, free);
		SG_SPRINT("sparse_vector[%d]=", i);
		for (index_t j=0; j<len; ++j)
		{
			SG_SPRINT("%d", vec[j].entry);
			if (j<len-1)
				SG_SPRINT(",");
		}

		SG_SPRINT("\n");

		for (index_t j=0; j<len; ++j)
			ASSERT(
					vec[j].entry==data.matrix[features->subset_idx_conversion( i)*num_vectors+j]);

		/* not necessary since feature matrix is in memory. for documentation */
		features->free_sparse_feature_vector(vec, i, free);
	}

	/* remove features subset */
	SG_SPRINT("\n-------------------\n"
	"removing subset from features\n"
	"-------------------\n");
	features->remove_subset();

	/* do some stuff do check and output */
	ASSERT(features->get_num_vectors()==num_vectors);
	SG_SPRINT("features->get_num_vectors(): %d\n", features->get_num_vectors());

	/* check get_Transposed method */
	SG_SPRINT("checking transpose...");
	check_transposed(features);
	SG_SPRINT("does work\n");

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		int32_t len;
		bool free;
		SGSparseVectorEntry<int32_t>* vec=features->get_sparse_feature_vector(i,
				len, free);
		SG_SPRINT("sparse_vector[%d]=", i);
		for (index_t j=0; j<len; ++j)
		{
			SG_SPRINT("%d", vec[j].entry);
			if (j<len-1)
				SG_SPRINT(",");
		}

		SG_SPRINT("\n");

		for (index_t j=0; j<len; ++j)
			ASSERT(vec[j].entry==data.matrix[i*num_vectors+j]);

		/* not necessary since feature matrix is in memory. for documentation */
		features->free_sparse_feature_vector(vec, i, free);
	}

	SG_UNREF(features);
	SG_FREE(data.matrix);

	SG_SPRINT("\nEND\n");
	exit_shogun();

	return 0;
}
