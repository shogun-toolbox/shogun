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

#ifndef MOMEMTUMCORRECTION_H
#define MOMEMTUMCORRECTION_H
#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/optimization/MinimizerContext.h>
#include <shogun/optimization/DescendCorrection.h>
namespace shogun
{
/** @brief This is a base class for momentum correction methods.
 *
 * The interfact will be used in DescendUpdaterWithCorrection::update_variable().
 *
 * An example of descend based correction is NesterovMomentumCorrection 
 */
class MomentumCorrection: public DescendCorrection
{
public:

	/*  Constructor */
	MomentumCorrection()
		:DescendCorrection()
	{
		init();
	}

	/*  Destructor */
	virtual ~MomentumCorrection() {};

	/*  Is the m_previous_descend_direction initialized?
	 *
	 *  @return whether m_previous_descend_direction is initialized
	 */
	virtual bool is_initialized()
	{
		return m_previous_descend_direction.vlen>0;
	}

	/*  Initialize m_previous_descend_direction?
	 *
	 *  @return len the length of m_previous_descend_direction to be initialized
	 */
	virtual void initialize_previous_direction(index_t len)
	{
		REQUIRE(len>0, "The length (%d) must be positive\n", len);
		m_previous_descend_direction=SGVector<float64_t>(len);
		m_previous_descend_direction.set_const(0.0);
	}

	/** Get corrected descend direction
	 *
	 * @param negative_descend_direction the negative descend direction
	 * @param idx the index of the direction
	 * @return DescendPair (corrected descend direction and the change to correct descend direction)

	 */
	virtual DescendPair get_corrected_descend_direction(float64_t negative_descend_direction,
		index_t idx)=0;

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
		SGVector<float64_t> value(m_previous_descend_direction.vlen);
		std::copy(m_previous_descend_direction.vector,
			m_previous_descend_direction.vector+m_previous_descend_direction.vlen,
			value.vector);
		std::string key="MomentumCorrection::m_previous_descend_direction";
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
		std::string key="MomentumCorrection::m_previous_descend_direction";
		SGVector<float64_t> value=context->get_data_sgvector_float64(key);
		m_previous_descend_direction=SGVector<float64_t>(value.vlen);
		std::copy(value.vector, value.vector+value.vlen,
			m_previous_descend_direction.vector);
	}

	/** Get the previous descend direction (velocity) given the index
	 *
	 * @param idx, index of the previous descend direction
	 *
	 * @return the previous descend direction
	 */
	virtual float64_t get_previous_descend_direction(index_t idx)
	{
		REQUIRE(idx>=0 && idx<m_previous_descend_direction.vlen,
			"Index (%d) is invalid\n", idx);
		return m_previous_descend_direction[idx];
	}

	/** Get the length of the previous descend direction (velocity)
	 *
	 * @return the length of the previous descend direction
	 */
	virtual float64_t get_length_previous_descend_direction()
	{
		return m_previous_descend_direction.vlen;
	}
protected:
	/*  used in momentum methods */
	SGVector<float64_t> m_previous_descend_direction;

private:
	/* Init */
	void init()
	{
		m_previous_descend_direction=SGVector<float64_t>();
	}
};

}
#endif
