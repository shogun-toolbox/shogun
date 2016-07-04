/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2014 - 2016 Soumyajit De
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

#include <shogun/io/SGIO.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/mathematics/Math.h>
#include <shogun/kernel/ShiftInvariantKernel.h>
#include <shogun/distance/CustomDistance.h>
#include <shogun/statistical_testing/MMD.h>
#include <shogun/statistical_testing/internals/KernelManager.h>
#include <shogun/statistical_testing/internals/mmd/MultiKernelMMD.h>

using namespace shogun;
using namespace internal;
using namespace mmd;

struct MultiKernelMMD::terms_t
{
	std::array<float64_t, 3> term{};
	std::array<float64_t, 3> diag{};
};

MultiKernelMMD::MultiKernelMMD(index_t nx, index_t ny, EStatisticType stype) : n_x(nx), n_y(ny), s_type(stype)
{
	SG_SDEBUG("number of samples are %d and %d!\n", n_x, n_y);
}

void MultiKernelMMD::set_distance(CCustomDistance* distance)
{
	m_distance=std::shared_ptr<CCustomDistance>(distance);
}

void MultiKernelMMD::add_term(terms_t& t, float32_t val, index_t i, index_t j) const
{
	if (i<n_x && j<n_x && i>=j)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_0!\n", i, j, val);
		t.term[0]+=val;
		if (i==j)
			t.diag[0]+=val;
	}
	else if (i>=n_x && j>=n_x && i>=j)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_1!\n", i, j, val);
		t.term[1]+=val;
		if (i==j)
			t.diag[1]+=val;
	}
	else if (i>=n_x && j<n_x)
	{
		SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_2!\n", i, j, val);
		t.term[2]+=val;
		if (i-n_x==j)
			t.diag[2]+=val;
	}
}

float64_t MultiKernelMMD::compute_mmd(terms_t& t) const
{
	t.term[0]=2*(t.term[0]-t.diag[0]);
	t.term[1]=2*(t.term[1]-t.diag[1]);
	SG_SDEBUG("term_0 sum (without diagonal) = %f!\n", t.term[0]);
	SG_SDEBUG("term_1 sum (without diagonal) = %f!\n", t.term[1]);
	if (s_type!=ST_BIASED_FULL)
	{
		t.term[0]/=n_x*(n_x-1);
		t.term[1]/=n_y*(n_y-1);
	}
	else
	{
		t.term[0]+=t.diag[0];
		t.term[1]+=t.diag[1];
		SG_SDEBUG("term_0 sum (with diagonal) = %f!\n", t.term[0]);
		SG_SDEBUG("term_1 sum (with diagonal) = %f!\n", t.term[1]);
		t.term[0]/=n_x*n_x;
		t.term[1]/=n_y*n_y;
	}
	SG_SDEBUG("term_0 (normalized) = %f!\n", t.term[0]);
	SG_SDEBUG("term_1 (normalized) = %f!\n", t.term[1]);

	SG_SDEBUG("term_2 sum (with diagonal) = %f!\n", t.term[2]);
	if (s_type==ST_UNBIASED_INCOMPLETE)
	{
		t.term[2]-=t.diag[2];
		SG_SDEBUG("term_2 sum (without diagonal) = %f!\n", t.term[2]);
		t.term[2]/=n_x*(n_x-1);
	}
	else
		t.term[2]/=n_x*n_y;
	SG_SDEBUG("term_2 (normalized) = %f!\n", t.term[2]);

	auto result=t.term[0]+t.term[1]-2*t.term[2];
	SG_SDEBUG("result = %f!\n", result);
	return result;
}

SGVector<float64_t> MultiKernelMMD::operator()(const KernelManager& kernel_mgr) const
{
	SG_SDEBUG("Entering!\n");
	REQUIRE(m_distance, "Distance instace is not set!\n");
	kernel_mgr.set_precomputed_distance(m_distance.get());

	SGVector<float64_t> result(kernel_mgr.num_kernels());
	std::vector<terms_t> terms(result.size());

	for (auto j=0; j<n_x+n_y; ++j)
	{
		for (auto i=j; i<n_x+n_y; ++i)
		{
			for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
			{
				auto kernel=kernel_mgr.kernel_at(k)->kernel(i, j);
				add_term(terms[k], kernel, i, j);
			}
		}
	}

	for (size_t k=0; k<kernel_mgr.num_kernels(); ++k)
	{
		result[k]=compute_mmd(terms[k]);
		SG_SDEBUG("result[%d] = %f!\n", k, result[k]);
	}
	terms.resize(0);

	kernel_mgr.unset_precomputed_distance();
	SG_SDEBUG("Leaving!\n");
	return result;
}
