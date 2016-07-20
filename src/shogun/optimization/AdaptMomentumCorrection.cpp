/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2015 Wu Lin
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
 *
 */

#include <shogun/optimization/AdaptMomentumCorrection.h>
#include <shogun/base/Parameter.h>
using namespace shogun;

void AdaptMomentumCorrection::set_momentum_correction(MomentumCorrection* correction)
{
	REQUIRE(correction,"MomentumCorrection must not NULL\n");
	REQUIRE(correction != this, "MomentumCorrection can not be itself\n");
	SG_REF(m_momentum_correction);
	m_momentum_correction=correction;
}

AdaptMomentumCorrection::~AdaptMomentumCorrection()
{
	SG_UNREF(m_momentum_correction);
};

bool AdaptMomentumCorrection::is_initialized()
{
	REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
	return m_momentum_correction->is_initialized();
}

void AdaptMomentumCorrection::set_correction_weight(float64_t weight)
{
	REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
	m_momentum_correction->set_correction_weight(weight);
}

void AdaptMomentumCorrection::initialize_previous_direction(index_t len)
{
	REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
	m_momentum_correction->initialize_previous_direction(len);
}

DescendPair AdaptMomentumCorrection::get_corrected_descend_direction(float64_t negative_descend_direction,
	index_t idx)
{
	REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
	REQUIRE(m_adapt_rate>0 && m_adapt_rate<1.0,"adaptive rate is invalid\n");
	index_t len=m_momentum_correction->get_length_previous_descend_direction();
	REQUIRE(idx>=0 && idx<len,
		"The index (%d) is invalid\n", idx);
		if(m_descend_rate.vlen==0)
		{
			m_descend_rate=SGVector<float64_t>(len);
			m_descend_rate.set_const(m_init_descend_rate);
		}
		float64_t rate=m_descend_rate[idx];
		float64_t pre=m_momentum_correction->get_previous_descend_direction(idx);
		DescendPair pair=m_momentum_correction->get_corrected_descend_direction(rate*negative_descend_direction, idx);
		float64_t cur=m_momentum_correction->get_previous_descend_direction(idx);
		if(pre*cur>0.0)
			rate*=(1.0+m_adapt_rate);
		else
		{
			if(pre*cur==0.0 && (cur>0.0 || pre>0.0))
				rate*=(1.0+m_adapt_rate);
			else
				rate*=(1.0-m_adapt_rate);
		}
		if(rate<m_rate_min)
			rate=m_rate_min;
		if(rate>m_rate_max)
			rate=m_rate_max;
		m_descend_rate[idx]=rate;
		return pair;
	}


void AdaptMomentumCorrection::set_adapt_rate(float64_t adapt_rate,
	float64_t rate_min, float64_t rate_max)
{
	REQUIRE(adapt_rate>0.0 && adapt_rate<1.0, "Adaptive rate (%f) must in (0,1)\n", adapt_rate);
	REQUIRE(rate_min>=0, "Minimum speedup rate (%f) must be non-negative\n", rate_min);
	REQUIRE(rate_max>rate_min, "Maximum speedup rate (%f) must greater than minimum speedup rate (%f)\n",
		rate_max, rate_min);
	m_adapt_rate=adapt_rate;
	m_rate_min=rate_min;
	m_rate_max=rate_max;
}

void AdaptMomentumCorrection::set_init_descend_rate(float64_t init_descend_rate)
{
	REQUIRE(init_descend_rate>0,"Init speedup rate (%f) must be positive\n", init_descend_rate);
	m_init_descend_rate=init_descend_rate;
}


void AdaptMomentumCorrection::init()
{
	m_momentum_correction=NULL;
	m_descend_rate=SGVector<float64_t>();
	m_adapt_rate=0;
	m_rate_min=0;
	m_rate_max=CMath::INFTY;
	m_init_descend_rate=1.0;

	SG_ADD(&m_adapt_rate, "AdaptMomentumCorrection__m_adapt_rate",
		"m_adapt_rate in AdaptMomentumCorrection", MS_NOT_AVAILABLE);
	SG_ADD(&m_rate_min, "AdaptMomentumCorrection__m_rate_min",
		"m_rate_min in AdaptMomentumCorrection", MS_NOT_AVAILABLE);
	SG_ADD(&m_rate_max, "AdaptMomentumCorrection__m_rate_max",
		"m_rate_max in AdaptMomentumCorrection", MS_NOT_AVAILABLE);
	SG_ADD(&m_init_descend_rate, "AdaptMomentumCorrection__m_init_descend_rate",
		"m_init_descend_rate in AdaptMomentumCorrection", MS_NOT_AVAILABLE);
	SG_ADD(&m_descend_rate, "AdaptMomentumCorrection__m_descend_rate",
		"m_descend_rate in AdaptMomentumCorrection", MS_NOT_AVAILABLE);
	SG_ADD((CSGObject **)&m_momentum_correction, "AdaptMomentumCorrection__m_momentum_correction",
		"m_momentum_correction in AdaptMomentumCorrection", MS_NOT_AVAILABLE);
}
