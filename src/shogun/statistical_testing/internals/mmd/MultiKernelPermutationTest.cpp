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
#include <shogun/kernel/Kernel.h>
#include <shogun/features/SubsetStack.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/BiasedFull.h>
#include <shogun/statistical_testing/internals/mmd/MultiKernelPermutationTest.h>

// TODO remove
#include <shogun/mathematics/eigen3.h>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::Map;
using std::cout;
using std::endl;
// TODO remove

using namespace shogun;
using namespace internal;
using namespace mmd;

struct MultiKernelPermutationTest::terms_t
{
	std::array<float64_t, 3> term{};
	std::array<float64_t, 3> diag{};
};

MultiKernelPermutationTest::MultiKernelPermutationTest(index_t nx, index_t ny, index_t nns, EStatisticType type)
: n_x(nx), n_y(ny), num_null_samples(nns), stype(type)
{
	SG_SDEBUG("number of samples are %d and %d!\n", n_x, n_y);
	SG_SDEBUG("Number of null samples is %d!\n", num_null_samples);
	permuted_inds=SGVector<index_t>(n_x+n_y);
	inverted_permuted_inds.resize(num_null_samples);
	for (auto i=0; i<num_null_samples; ++i)
		inverted_permuted_inds[i].resize(permuted_inds.vlen);
}

MultiKernelPermutationTest::~MultiKernelPermutationTest()
{
}

void MultiKernelPermutationTest::add_term(terms_t& terms, float64_t val, index_t i, index_t j)
{
	if (i<n_x && j<n_x && i>=j)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_0!\n", i, j, val);
		terms.term[0]+=val;
		if (i==j)
			terms.diag[0]+=val;
	}
	else if (i>=n_x && j>=n_x && i>=j)
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

float64_t MultiKernelPermutationTest::compute_mmd(terms_t& terms)
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

SGVector<bool> MultiKernelPermutationTest::operator()(const KernelManager& kernel_mgr)
{
	SG_SDEBUG("Entering!\n");

	for (auto n=0; n<num_null_samples; ++n)
	{
		std::iota(permuted_inds.vector, permuted_inds.vector+permuted_inds.vlen, 0);
		CMath::permute(permuted_inds);
		for (auto i=0; i<permuted_inds.vlen; ++i)
			inverted_permuted_inds[n][permuted_inds[i]]=i;
	}

	SGVector<float64_t> statistic(kernel_mgr.num_kernels());
	std::vector<terms_t> stat_terms(kernel_mgr.num_kernels());
	for (auto col=0; col<n_x+n_y; ++col)
	{
		for (auto row=col; row<n_x+n_y; ++row)
		{
			for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
			{
				float64_t kernel_value=kernel_mgr.kernel_at(k)->kernel(row, col);
				add_term(stat_terms[k], kernel_value, row, col);
			}
		}
	}
	for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
		statistic[k]=compute_mmd(stat_terms[k]);
	stat_terms.resize(0);

	SGMatrix<float64_t> null_samples(kernel_mgr.num_kernels(), num_null_samples);
	Map<MatrixXd> null_samples_map(null_samples.data(), null_samples.num_rows, null_samples.num_cols);
#pragma omp parallel for
	for (auto n=0; n<num_null_samples; ++n)
	{
		std::vector<terms_t> terms(kernel_mgr.num_kernels());
		for (auto col=0; col<n_x+n_y; ++col)
		{
			for (auto row=col; row<n_x+n_y; ++row)
			{
				auto row_inds_inv=inverted_permuted_inds[n][row];
				auto col_inds_inv=inverted_permuted_inds[n][col];
				for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
				{
					float64_t kernel_value=kernel_mgr.kernel_at(k)->kernel(row, col);
					add_term(terms[k], kernel_value, row_inds_inv, col_inds_inv);
				}
			}
		}
		for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
			null_samples(k, n)=compute_mmd(terms[k]);
	}

	SGVector<bool> result(kernel_mgr.num_kernels());
	MatrixXd transposed_null_samples=null_samples_map.transpose();
	for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
	{
		SGVector<float64_t> null_samples_k(transposed_null_samples.col(k).data(), num_null_samples, false);
		std::sort(null_samples_k.data(), null_samples_k.data()+null_samples_k.size());
		SG_SDEBUG("statistic=%f\n", statistic[k]);
		float64_t idx=null_samples_k.find_position_to_insert(statistic[k]);
		SG_SDEBUG("index=%f\n", idx);
		auto p_value=1.0-idx/num_null_samples;
		bool rejected=p_value<alpha;
		SG_SDEBUG("p-value=%f, alpha=%f, rejected=%d\n", p_value, alpha, rejected);
		result[k]=rejected;
	}

	SG_SDEBUG("Leaving!\n");
	return result;
}

void MultiKernelPermutationTest::set_alpha(float64_t alp)
{
	alpha=alp;
}
