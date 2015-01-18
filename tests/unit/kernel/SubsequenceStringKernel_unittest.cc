/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2014 Soumyajit De
 */

#include <shogun/lib/common.h>
#include <shogun/lib/SGString.h>
#include <shogun/lib/SGStringList.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/kernel/string/SubsequenceStringKernel.h>
#include <gtest/gtest.h>

#ifdef HAVE_EIGEN3
#include <shogun/mathematics/eigen3.h>

using namespace Eigen;

#endif // HAVE_EIGEN3

using namespace shogun;

TEST(SubsequenceStringKernel, compute)
{
	const index_t num_strings=2;
	const index_t len=7;

	const char* doc_1="ABCDEFG";
	const char* doc_2="EFGHIJK";

	SGString<char> string_1(len);
	for (index_t i=0; i<len; i++)
		string_1.string[i] = doc_1[i];

	SGString<char> string_2(len);
	for (index_t i=0; i<len; i++)
		string_2.string[i] = doc_2[i];

	SGStringList<char> list(num_strings, len);
	list.strings[0]=string_1;
	list.strings[1]=string_2;

	CStringFeatures<char>* s_feats=new CStringFeatures<char>(list, ALPHANUM);

	// create string subsequence kernel with max subsequence length 2 and
	// a decay factor (lambda) 1.0 (no decay)
	CSubsequenceStringKernel* kernel=new CSubsequenceStringKernel(s_feats, s_feats, 2, 1);
	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix();

	EXPECT_NEAR(kernel_matrix(0,0), 1.0, 1E-10);
	EXPECT_NEAR(kernel_matrix(1,1), 1.0, 1E-10);
	EXPECT_NEAR(kernel_matrix(0,1), 0.214285714285714246, 1E-10);
	EXPECT_NEAR(kernel_matrix(1,0), 0.214285714285714246, 1E-10);

	SG_UNREF(kernel);
}

#ifdef HAVE_EIGEN3
TEST(SubsequenceStringKernel, psd_random_feat)
{
	const index_t num_strings=10;
	const index_t max_len=20;
	const index_t min_len=max_len/2;

	SGStringList<char> list(num_strings, max_len);
	for (index_t i=0; i<num_strings; ++i)
	{
		index_t cur_len=CMath::random(min_len, max_len);
		SGString<char> str(cur_len);
		for (index_t l=0; l<cur_len; ++l)
			str.string[l]=char(CMath::random('A','Z'));
		list.strings[i]=str;
	}

	CStringFeatures<char>* s_feats=new CStringFeatures<char>(list, ALPHANUM);
	int32_t s_len=CMath::random(1, min_len);
	float64_t lambda=CMath::random(0.0, 1.0);
	CSubsequenceStringKernel* kernel=new CSubsequenceStringKernel(s_feats, s_feats, s_len, lambda);

	SGMatrix<float64_t> kernel_matrix=kernel->get_kernel_matrix();
	Map<MatrixXd> km_map(kernel_matrix.matrix, kernel_matrix.num_rows, kernel_matrix.num_cols);

	VectorXcd eig=km_map.eigenvalues();
	for (index_t i=0; i<eig.size(); ++i)
		EXPECT_GE(eig[i].real(), 0.0);

	SG_UNREF(kernel);
}
#endif // HAVE_EIGEN3
