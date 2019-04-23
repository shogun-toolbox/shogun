/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn, Thoralf Klein, 
 *          Viktor Gal
 */

#include <shogun/base/init.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/Subset.h>

using namespace shogun;

void test()
{

	index_t num_vectors=10;
	index_t num_dimensions=7;
	index_t num_features=3;

	/* create some sparse data */
	SGSparseMatrix<float64_t> data=SGSparseMatrix<float64_t>(num_dimensions,
			num_vectors);

	for (index_t i=0; i<num_vectors; ++i)
	{
		/* put elements only at even indices */
		data.sparse_matrix[i]=SGSparseVector<float64_t>(num_features);

		/* fill */
		for (index_t j=0; j<num_features; ++j)
		{
			data.sparse_matrix[i].features[j].entry=i+j;
			data.sparse_matrix[i].features[j].feat_index=3*j;
		}
	}
	auto f=std::make_shared<SparseFeatures<float64_t>>(data);

	/* display sparse matrix */
	SG_SPRINT("original data\n");
	for (index_t i=0; i<num_vectors; ++i)
	{
		SG_SPRINT("sparse vector at %i: [", i);
		for (index_t j=0; j<num_features; ++j)
			SG_SPRINT("%f, ", data.sparse_matrix[i].features[j].entry);

		SG_SPRINT("]\n");
	}

	/* indices for a subset */
	index_t offset_subset=1;
	SGVector<index_t> feature_subset(8);
	SGVector<index_t>::range_fill_vector(feature_subset.vector, feature_subset.vlen,
			offset_subset);
	SGVector<index_t>::display_vector(feature_subset.vector, feature_subset.vlen,
			"feature subset");

	/* set subset and print data */
	f->add_subset(feature_subset);
	SG_SPRINT("feature vectors after setting subset on original data:\n");
	for (index_t i=0; i<f->get_num_vectors(); ++i)
	{
		SGSparseVector<float64_t> vec=f->get_sparse_feature_vector(i);
		SG_SPRINT("sparse vector at %i: ", i);
		for (index_t j=0; j<num_features; ++j)
			SG_SPRINT("%f, ", vec.features[j].entry);

		SG_SPRINT("]\n");
		f->free_sparse_feature_vector(i);
	}

	/* indices that are to copy */
	index_t offset_copy=2;
	SGVector<index_t> feature_copy_subset(4);
	SGVector<index_t>::range_fill_vector(feature_copy_subset.vector,
			feature_copy_subset.vlen, offset_copy);
	SGVector<index_t>::display_vector(feature_copy_subset.vector, feature_copy_subset.vlen,
			"indices that are to be copied");

	/* copy a subset of features */
	auto subset_copy=
			f->copy_subset(feature_copy_subset)->as<SparseFeatures<float64_t>>();

	/* print copied subset */
	SG_SPRINT("copied features:\n");
	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGSparseVector<float64_t> vec=subset_copy->get_sparse_feature_vector(i);
		SG_SPRINT("sparse vector at %i: ", i);
		for (index_t j=0; j<num_features; ++j)
			SG_SPRINT("%f, ", vec.features[j].entry);

		SG_SPRINT("]\n");
		subset_copy->free_sparse_feature_vector(i);
	}

	/* test if all elements are copied correctly */
	for (index_t i=0; i<subset_copy->get_num_vectors(); ++i)
	{
		SGSparseVector<float64_t> vec=subset_copy->get_sparse_feature_vector(i);
		index_t ind=i+offset_copy+offset_subset+1;

		for (index_t j=0; j<vec.num_feat_entries; ++j)
		{
			float64_t a_entry=vec.features[j].entry;
			float64_t b_entry=data.sparse_matrix[ind].features[j].entry;
			index_t a_idx=vec.features[j].feat_index;
			index_t b_idx=data.sparse_matrix[ind].features[j].feat_index;

			ASSERT(a_entry==b_entry);
			ASSERT(a_idx==b_idx);
		}

		subset_copy->free_sparse_feature_vector(i);
	}

}
int main(int argc, char **argv)
{
	init_shogun_with_defaults();

	test();

	exit_shogun();

	return 0;
}

