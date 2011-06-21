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
#include <shogun/features/SimpleFeatures.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

const int32_t num_vectors=6;
const int32_t dim_features=6;

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);
	const int32_t num_subset_idx=CMath::random(1, num_vectors);

	/* alloc feature data matrix */
	SGMatrix<int32_t> data(
			(int32_t*) SG_MALLOC(sizeof(int32_t)*num_vectors*dim_features),
			dim_features, num_vectors);

	/* fill matrix with random data */
	for (index_t i=0; i<num_vectors; ++i)
	{
		for (index_t j=0; j<dim_features; ++j)
			data.matrix[i*dim_features+j]=CMath::random(-5, 5);
	}

	/* create simple features */
	CSimpleFeatures<int32_t>* features=new CSimpleFeatures<int32_t> (data);
	SG_REF(features);

	/* print feature matrix */
	CMath::display_matrix(data.matrix, data.num_rows, data.num_cols,
			"feature matrix");

	/* create subset indices with stratified corss validaiton splitting class */
	SGVector<index_t> subset_idx(CMath::randperm(num_subset_idx),
			num_subset_idx);

	/* print subset indices */
	CMath::display_vector(subset_idx.vector, subset_idx.vlen, "subset indices");

	/* apply subset to features */
	SG_SPRINT("\n\n-------------------\n"
			"applying subset to features\n"
			"-------------------\n");
	features->set_subset(subset_idx);

	/* do some stuff do check and output */
	ASSERT(features->get_num_vectors()==num_subset_idx);

	SG_SPRINT("features->get_num_vectors(): %d\n", features->get_num_vectors());

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_feature_vector(i);
		SG_SPRINT("vector %d: ", i);
		CMath::display_vector(vec.vector, vec.vlen);

		for (index_t j=0; j<dim_features; ++j)
			ASSERT(vec.vector[j]==data.matrix[features->subset_idx_conversion(
					i)*num_vectors+j]);
	}

	/* remove features subset */
	SG_SPRINT("\n\n-------------------\n"
			"removing subset from features\n"
			"-------------------\n");
	features->remove_subset();

	/* do some stuff do check and output */
	ASSERT(features->get_num_vectors()==num_vectors);
	SG_SPRINT("features->get_num_vectors(): %d\n", features->get_num_vectors());

	for (index_t i=0; i<features->get_num_vectors(); ++i)
	{
		SGVector<int32_t> vec=features->get_feature_vector(i);
		SG_SPRINT("vector %d: ", i);
		CMath::display_vector(vec.vector, vec.vlen);

		for (index_t j=0; j<dim_features; ++j)
			ASSERT(vec.vector[j]==data.matrix[features->subset_idx_conversion(i)
					*num_vectors+j]);
	}


	SG_UNREF(features);

	SG_SPRINT("\nEND\n");
	exit_shogun();

	return 0;
}

