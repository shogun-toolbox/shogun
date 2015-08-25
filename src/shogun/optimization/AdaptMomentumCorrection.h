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

#ifndef ADAPTMOMEMTUMCORRECTION_H
#define ADAPTMOMEMTUMCORRECTION_H
#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/optimization/MomentumCorrection.h>
namespace shogun
{
/** @brief This implements the adaptive momentum correction method.
 *
 */
class AdaptMomentumCorrection: public MomentumCorrection
{
public:

	/*  Constructor */
	AdaptMomentumCorrection()
		:MomentumCorrection()
	{
		init();
	}

	virtual void set_momentum_correction(MomentumCorrection* correction)
	{
		REQUIRE(correction,"MomentumCorrection must not NULL\n");
		REQUIRE(correction != this, "MomentumCorrection can be itself\n");
		m_momentum_correction=correction;
	}

	/*  Destructor */
	virtual ~AdaptMomentumCorrection() { m_momentum_correction=NULL; };

	/*  Is the m_previous_descend_direction initialized?
	 *
	 *  @return whether m_previous_descend_direction is initialized
	 */
	virtual bool is_initialized()
	{
		REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
		return m_momentum_correction->is_initialized();
	}

	virtual void set_correction_weight(float64_t weight)
	{
		REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
		m_momentum_correction->set_correction_weight(weight);
	}

	/*  Initialize m_previous_descend_direction?
	 *
	 *  @return len the length of m_previous_descend_direction to be initialized
	 */
	virtual void initialize_previous_direction(index_t len)
	{
		REQUIRE(m_momentum_correction,"MomentumCorrection must set\n");
		m_momentum_correction->initialize_previous_direction(len);
	}

	/** Get corrected descend direction
	 *
	 * @param negative_descend_direction the negative descend direction
	 * @param idx the index of the direction
	 * @return DescendPair (corrected descend direction and the change to correct descend direction)
	*/
	virtual DescendPair get_corrected_descend_direction(float64_t negative_descend_direction,
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

	/** Update a context object to store mutable variables
	 * used in descend update
	 *
	 * This method will be called by
	 * DescendUpdaterWithCorrection::update_context()
	 *
	 * @param context, a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "context must set\n");
		SGVector<float64_t> value(m_descend_rate.vlen);
		std::copy(m_descend_rate.vector,
			m_descend_rate.vector+m_descend_rate.vlen,
			value.vector);
		std::string key="AdaptMomentumCorrection::m_descend_rate";
		context->save_data(key, value);
	}

	/** Load the given context object to restore mutable variables
	 *
	 * This method will be called by
	 * DescendUpdaterWithCorrection::load_from_context(CMinimizerContext* context)
	 *
	 * @param context, a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "context must set\n");
		std::string key="AdaptMomentumCorrection::m_descend_rate";
		SGVector<float64_t> value=context->get_SGVector_float64(key);
		m_descend_rate=SGVector<float64_t>(value.vlen);
		std::copy(value.vector, value.vector+value.vlen,
			m_descend_rate.vector);
	}

	virtual void set_adapt_rate(float64_t adapt_rate, float64_t rate_min=0.0, float64_t rate_max=CMath::INFTY)
	{
		REQUIRE(adapt_rate>0.0 && adapt_rate<1.0, "Adaptive rate (%f) must in (0,1)\n", adapt_rate);
		REQUIRE(rate_min>=0, "Minimum speedup rate (%f) must be non-negative\n", rate_min);
		REQUIRE(rate_max>rate_min, "Maximum speedup rate (%f) must greater than minimum speedup rate (%f)\n",
			rate_max, rate_min);
		m_adapt_rate=adapt_rate;
		m_rate_min=rate_min;
		m_rate_max=rate_max;
	}

	virtual void set_init_descend_rate(float64_t init_descend_rate)
	{
		REQUIRE(init_descend_rate>0,"Init speedup rate (%f) must be positive\n", init_descend_rate);
		m_init_descend_rate=init_descend_rate;
	}
protected:
	SGVector<float64_t> m_descend_rate;
	MomentumCorrection* m_momentum_correction;
	float64_t m_adapt_rate;
	float64_t m_rate_min;
	float64_t m_rate_max;
	float64_t m_init_descend_rate;
private:
	/* Init */
	void init()
	{
		m_momentum_correction=NULL;
		m_descend_rate=SGVector<float64_t>();
		m_adapt_rate=0;
		m_rate_min=0;
		m_rate_max=CMath::INFTY;
		m_init_descend_rate=1.0;
	}
};

}
#endif
