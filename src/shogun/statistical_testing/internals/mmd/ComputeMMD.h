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

#ifndef COMPUTE_MMD_H_
#define COMPUTE_MMD_H_

#include <array>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/statistical_testing/MMD.h>

namespace shogun
{

namespace internal
{

namespace mmd
{

struct terms_t
{
	std::array<float64_t, 3> term{};
	std::array<float64_t, 3> diag{};
};

/**
 * @brief Class Compute blah blah.
 */
struct ComputeMMD
{
	ComputeMMD(index_t n_x, index_t n_y, EStatisticType stype) : m_n_x(n_x), m_n_y(n_y), m_stype(stype)
	{
		SG_SDEBUG("number of samples are %d and %d!\n", n_x, n_y);
	}

	template <typename T>
	inline void add_term(terms_t& terms, T val, index_t i, index_t j) const
	{
		if (i<m_n_x && j<m_n_x && i>=j)
		{
			SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_0!\n", i, j, val);
			terms.term[0]+=val;
			if (i==j)
				terms.diag[0]+=val;
		}
		else if (i>=m_n_x && j>=m_n_x && i>=j)
		{
			SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_1!\n", i, j, val);
			terms.term[1]+=val;
			if (i==j)
				terms.diag[1]+=val;
		}
		else if (i>=m_n_x && j<m_n_x)
		{
			SG_SDEBUG("Adding Kernel(%d,%d)=%f to term_2!\n", i, j, val);
			terms.term[2]+=val;
			if (i-m_n_x==j)
				terms.diag[2]+=val;
		}
	}

	inline float64_t compute(terms_t& terms) const
	{
		terms.term[0]=2*(terms.term[0]-terms.diag[0]);
		terms.term[1]=2*(terms.term[1]-terms.diag[1]);
		SG_SDEBUG("term_0 sum (without diagonal) = %f!\n", terms.term[0]);
		SG_SDEBUG("term_1 sum (without diagonal) = %f!\n", terms.term[1]);
		if (m_stype!=EStatisticType::ST_BIASED_FULL)
		{
			terms.term[0]/=m_n_x*(m_n_x-1);
			terms.term[1]/=m_n_y*(m_n_y-1);
		}
		else
		{
			terms.term[0]+=terms.diag[0];
			terms.term[1]+=terms.diag[1];
			SG_SDEBUG("term_0 sum (with diagonal) = %f!\n", terms.term[0]);
			SG_SDEBUG("term_1 sum (with diagonal) = %f!\n", terms.term[1]);
			terms.term[0]/=m_n_x*m_n_x;
			terms.term[1]/=m_n_y*m_n_y;
		}
		SG_SDEBUG("term_0 (normalized) = %f!\n", terms.term[0]);
		SG_SDEBUG("term_1 (normalized) = %f!\n", terms.term[1]);

		SG_SDEBUG("term_2 sum (with diagonal) = %f!\n", terms.term[2]);
		if (m_stype==EStatisticType::ST_UNBIASED_INCOMPLETE)
		{
			terms.term[2]-=terms.diag[2];
			SG_SDEBUG("term_2 sum (without diagonal) = %f!\n", terms.term[2]);
			terms.term[2]/=m_n_x*(m_n_x-1);
		}
		else
			terms.term[2]/=m_n_x*m_n_y;
		SG_SDEBUG("term_2 (normalized) = %f!\n", terms.term[2]);

		auto result=terms.term[0]+terms.term[1]-2*terms.term[2];
		SG_SDEBUG("result = %f!\n", result);
		return result;
	}

	const index_t m_n_x;
	const index_t m_n_y;
	const EStatisticType m_stype;
};

}

}

}
#endif // COMPUTE_MMD_H_
