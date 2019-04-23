/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Sergey Lisitsyn
 */

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/Subset.h>

using namespace shogun;

void test()
{

	SGMatrix<float64_t> data(3, 10);
	auto f=std::make_shared<DenseFeatures<float64_t>>(data);
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

	auto subset_copy=
			f->copy_subset(feature_copy_subset)->as<DenseFeatures<float64>>();

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

}

int main(int argc, char **argv)
{
	test();

	return 0;
}

