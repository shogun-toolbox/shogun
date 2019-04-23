/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Pan Deng, Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGVector.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/kernel/string/SubsequenceStringKernel.h>
#include <gtest/gtest.h>
#include <shogun/mathematics/eigen3.h>
#include <shogun/mathematics/UniformIntDistribution.h>
#include <shogun/mathematics/UniformRealDistribution.h>

#include <random>

using namespace Eigen;

using namespace shogun;

TEST(SubsequenceStringKernel, compute)
{
	const index_t num_strings=2;
	const index_t len=7;

	const char* doc_1="ABCDEFG";
	const char* doc_2="EFGHIJK";

	SGVector<char> string_1(len);
	for (index_t i=0; i<len; i++)
		string_1[i] = doc_1[i];

	SGVector<char> string_2(len);
	for (index_t i=0; i<len; i++)
		string_2[i] = doc_2[i];

	std::vector<SGVector<char>> list;
	list.reserve(num_strings);
	list.push_back(string_1);
	list.push_back(string_2);

	auto s_feats=std::make_shared<StringFeatures<char>>(list, ALPHANUM);

	// create string subsequence kernel with max subsequence length 2 and
	// a decay factor (lambda) 1.0 (no decay)
	auto kernel=std::make_shared<SubsequenceStringKernel>(s_feats, s_feats, 2, 1);
	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix();

	EXPECT_NEAR(kernel_matrix(0,0), 1.0, 1E-10);
	EXPECT_NEAR(kernel_matrix(1,1), 1.0, 1E-10);
	EXPECT_NEAR(kernel_matrix(0,1), 0.214285714285714246, 1E-10);
	EXPECT_NEAR(kernel_matrix(1,0), 0.214285714285714246, 1E-10);


}

TEST(SubsequenceStringKernel, psd_random_feat)
{
	const int32_t seed = 12;
	const index_t num_strings=10;
	const index_t max_len=20;
	const index_t min_len=max_len/2;

	std::mt19937_64 prng(seed);
	UniformIntDistribution<int32_t> uniform_int_dist;
	UniformRealDistribution<float64_t> uniform_real_dist;
	std::vector<SGVector<char>> list;
	list.reserve(num_strings);
	for (index_t i=0; i<num_strings; ++i)
	{
		index_t cur_len=uniform_int_dist(prng, {min_len, max_len});
		SGVector<char> str(cur_len);
		for (index_t l=0; l<cur_len; ++l)
			str[l]=char(uniform_int_dist(prng, {'A','Z'}));
		list.push_back(str);
	}

	auto s_feats=std::make_shared<StringFeatures<char>>(list, ALPHANUM);
	int32_t s_len=uniform_int_dist(prng, {1, min_len});
	float64_t lambda=uniform_real_dist(prng, {0.0, 1.0});
	auto kernel=std::make_shared<SubsequenceStringKernel>(s_feats, s_feats, s_len, lambda);

	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix();
	Map<MatrixXd> km_map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);

	VectorXcd eig=km_map.eigenvalues();
	for (index_t i=0; i<eig.size(); ++i)
		EXPECT_GE(eig[i].real(), 0.0);


}

