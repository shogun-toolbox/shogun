/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Soumyajit De
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 */

#include <shogun/base/some.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/evaluation/CrossValidationSplitting.h>
#include <shogun/statistical_testing/internals/mmd/PermutationTestCrossValidation.h>
#include <shogun/statistical_testing/internals/mmd/BiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/UnbiasedIncomplete.h>

// TODO remove
#include <shogun/mathematics/eigen3.h>
#include <iostream>

using Eigen::Matrix;
using Eigen::Dynamic;
using Eigen::Map;
using std::cout;
using std::endl;
// TODO remove

namespace shogun
{

namespace internal
{

namespace mmd
{

struct PermutationTestCrossValidation::terms_t
{
	std::array<float64_t, 3> term{};
	std::array<float64_t, 3> diag{};
};

PermutationTestCrossValidation::PermutationTestCrossValidation(index_t nx, index_t ny, index_t nns, EStatisticType type)
: n_x(nx), n_y(ny), num_null_samples(nns), stype(type)
{
	SG_SDEBUG("number of samples are %d and %d!\n", n_x, n_y);
	SG_SDEBUG("Number of null samples is %d!\n", num_null_samples);

}

PermutationTestCrossValidation::~PermutationTestCrossValidation()
{
}

template <typename T>
void PermutationTestCrossValidation::add_term(terms_t& terms, T val, index_t i, index_t j)
{
	if (i<n_x && j<n_x && i<=j)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_0!\n", i, j, val);
		terms.term[0]+=val;
		if (i==j)
			terms.diag[0]+=val;
	}
	else if (i>=n_x && j>=n_x && i<=j)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_1!\n", i, j, val);
		terms.term[1]+=val;
		if (i==j)
			terms.diag[1]+=val;
	}
	else if (i>=n_x && j<n_x)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_2!\n", i, j, val);
		terms.term[2]+=val;
		if (i-n_x==j)
			terms.diag[2]+=val;
	}
}

float64_t PermutationTestCrossValidation::compute_mmd(terms_t& terms)
{
	terms.term[0]=2*(terms.term[0]-terms.diag[0]);
	terms.term[1]=2*(terms.term[1]-terms.diag[1]);
	SG_SDEBUG("term_0 sum (without diagonal) = %f!\n", terms.term[0]);
	SG_SDEBUG("term_1 sum (without diagonal) = %f!\n", terms.term[1]);
	if (stype!=EStatisticType::ST_BIASED_FULL)
	{
		terms.term[0]/=n_x*(n_x-1);
		terms.term[1]/=n_y*(n_y-1);
	}
	else
	{
		terms.term[0]+=terms.diag[0];
		terms.term[1]+=terms.diag[1];
		SG_SDEBUG("term_0 sum (with diagonal) = %f!\n", terms.term[0]);
		SG_SDEBUG("term_1 sum (with diagonal) = %f!\n", terms.term[1]);
		terms.term[0]/=n_x*n_x;
		terms.term[1]/=n_y*n_y;
	}
	SG_SDEBUG("term_0 (normalized) = %f!\n", terms.term[0]);
	SG_SDEBUG("term_1 (normalized) = %f!\n", terms.term[1]);

	SG_SDEBUG("term_2 sum (with diagonal) = %f!\n", terms.term[2]);
	if (stype==EStatisticType::ST_UNBIASED_INCOMPLETE)
	{
		terms.term[2]-=terms.diag[2];
		SG_SDEBUG("term_2 sum (without diagonal) = %f!\n", terms.term[2]);
		terms.term[2]/=n_x*(n_x-1);
	}
	else
		terms.term[2]/=n_x*n_y;
	SG_SDEBUG("term_2 (normalized) = %f!\n", terms.term[2]);

	return terms.term[0]+terms.term[1]-2*terms.term[2];
}

template <typename T>
void PermutationTestCrossValidation::operator()(const SGMatrix<T>& km, index_t k)
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(rejections.num_rows==num_runs*num_folds,
		"Number of rows in the measure matrix (was %d), has to be >= %d*%d = %d!\n",
		rejections.num_rows, num_runs, num_folds, num_runs*num_folds);
	REQUIRE(k>=0 && k<rejections.num_cols,
		"Kernel index (%d) has to be between [0, %d]!\n",
		k, rejections.num_cols-1);

//	km.display_matrix("kernel_matrix");
//	typedef Matrix<T, Dynamic, Dynamic> MatrixXt;
//	Map<MatrixXt> map(km.data(), km.num_rows, km.num_cols);
//	cout << map << endl;

	SGVector<int64_t> dummy_labels_x(n_x);
	SGVector<int64_t> dummy_labels_y(n_y);
	auto kfold_x=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_x), num_folds);
	auto kfold_y=some<CCrossValidationSplitting>(new CBinaryLabels(dummy_labels_y), num_folds);

	for (auto i=0; i<num_runs; ++i)
	{
		kfold_x->build_subsets();
		kfold_y->build_subsets();
		for (auto j=0; j<num_folds; ++j)
		{
			SGVector<index_t> x_inds=kfold_x->generate_subset_inverse(j);
			SGVector<index_t> y_inds=kfold_y->generate_subset_inverse(j);
//			x_inds.display_vector("x_inds");
//			y_inds.display_vector("y_inds");
			std::for_each(y_inds.data(), y_inds.data()+y_inds.size(), [this](index_t& val) { val += n_x; });
			SGVector<index_t> xy_inds(x_inds.size()+y_inds.size());
			std::copy(x_inds.data(), x_inds.data()+x_inds.size(), xy_inds.data());
			std::copy(y_inds.data(), y_inds.data()+y_inds.size(), xy_inds.data()+x_inds.size());
//			xy_inds.display_vector("xy_inds");

			// compute statistic
			SGVector<index_t> inverted_inds(n_x+n_y);
			std::fill(inverted_inds.data(), inverted_inds.data()+n_x+n_y, -1);
			for (int idx=0; idx<xy_inds.size(); ++idx)
				inverted_inds[xy_inds[idx]]=idx;
			terms_t stat_terms;
			for (auto row=0; row<n_x+n_y; ++row)
			{
				for (auto col=0; col<n_x+n_y; ++col)
				{
					auto inverted_row=inverted_inds[row];
					auto inverted_col=inverted_inds[col];
					if (inverted_row!=-1 && inverted_col!=-1)
					{
						add_term(stat_terms, km(row, col), inverted_row, inverted_col);
					}
				}
			}
			auto statistic=compute_mmd(stat_terms);

			// compute the null samples
			SGVector<float64_t> null_samples(num_null_samples);
#pragma omp parallel for
			for (auto n=0; n<num_null_samples; ++n)
			{
				auto stack=some<CSubsetStack>();
				stack->add_subset(xy_inds);

				SGVector<index_t> permutation_inds(xy_inds.size());
				std::iota(permutation_inds.data(), permutation_inds.data()+permutation_inds.size(), 0);
				CMath::permute(permutation_inds);
				stack->add_subset(permutation_inds);

				SGVector<index_t> inds=stack->get_last_subset()->get_subset_idx();
//				inds.display_vector("inds (after permutation)");

				SGVector<index_t> inverted_permutation_inds(n_x+n_y);
				std::fill(inverted_permutation_inds.data(), inverted_permutation_inds.data()+n_x+n_y, -1);
				for (int idx=0; idx<inds.size(); ++idx)
					inverted_permutation_inds[inds[idx]]=idx;
//				inverted_permutation_inds.display_vector("inv permutation inds");

				terms_t terms;
				for (auto row=0; row<n_x+n_y; ++row)
				{
					for (auto col=0; col<n_x+n_y; ++col)
					{
						auto permuted_row=inverted_permutation_inds[row];
						auto permuted_col=inverted_permutation_inds[col];
						if (permuted_row!=-1 && permuted_col!=-1)
						{
							add_term(terms, km(row, col), permuted_row, permuted_col);
						}
					}
				}
				null_samples[n]=compute_mmd(terms);
			}
			std::sort(null_samples.data(), null_samples.data()+null_samples.size());
//			null_samples.display_vector("null_samples");
			SG_SDEBUG("statistic=%f\n", statistic);
			float64_t idx=null_samples.find_position_to_insert(statistic);
			auto p_value=1.0-idx/num_null_samples;
			SG_SDEBUG("p-value=%f, rejected=%d\n", p_value, p_value<alpha);
			rejections(i*num_folds+j, k)=p_value<alpha;
		}
	}

	SG_SDEBUG("Leaving!\n");
}

void PermutationTestCrossValidation::set_num_runs(index_t nr)
{
	num_runs=nr;
}

void PermutationTestCrossValidation::set_num_folds(index_t nf)
{
	num_folds=nf;
}

void PermutationTestCrossValidation::set_alpha(float64_t alp)
{
	alpha=alp;
}

void PermutationTestCrossValidation::set_measure_matrix(SGMatrix<float64_t> measures)
{
	rejections=measures;
}

template void PermutationTestCrossValidation::operator()<float64_t>(const SGMatrix<float64_t>& km, index_t k);
template void PermutationTestCrossValidation::add_term<float64_t>(terms_t& terms, float64_t val, index_t i, index_t j);

}

}

}
