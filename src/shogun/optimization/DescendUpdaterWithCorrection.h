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

#ifndef DESCENDUPDATERWITHCORRECTION_H
#define DESCENDUPDATERWITHCORRECTION_H
#include <shogun/lib/config.h>
#include <shogun/optimization/DescendUpdater.h>
#include <shogun/optimization/MomentumCorrection.h>

namespace shogun
{
/** @brief This is a base class for descend update with descend based correction.
 *
 * The class enables descend update with descend-based correction.
 *
 * Given a target variable, \f$w\f$, and its negative descend direction \f$g\f$,
 * the class will first correct the descend direction, \f$g\f$, and then update \f$w\f$
 * based on \f$g^{corrected}\f$ (eg, subtracting \f$g^{corrected}\f$)
 *
 * Note that an example of \f$d\f$ is to simply use the gradient wrt \f$w\f$. 
 * An example of using descend based correction can be found at StandardMomentumCorrection
 *
 */
class DescendUpdaterWithCorrection: public DescendUpdater
{
public:
	/*  Destructor */
	virtual ~DescendUpdaterWithCorrection() {};

	/** Update the target variable based on the given negative descend direction
	 *
	 * Note that this method will update the target variable in place.
	 * This method will be called by FirstOrderMinimizer::minimize()
	 * 
	 * @param variable_reference a reference of the target variable
	 * @param raw_negative_descend_direction the negative descend direction given the current value
	 * @param learning_rate learning rate
	 */
	virtual void update_variable(SGVector<float64_t> variable_reference,
		SGVector<float64_t> raw_negative_descend_direction, float64_t learning_rate)
	{
		REQUIRE(variable_reference.vlen>0,"variable_reference must set\n");
		REQUIRE(variable_reference.vlen==raw_negative_descend_direction.vlen,
			"The length of variable_reference (%d) and the length of gradient (%d) do not match\n",
			variable_reference.vlen,raw_negative_descend_direction.vlen);

		if(m_correction)
		{
			MomentumCorrection* momentum_correction=dynamic_cast<MomentumCorrection *>(m_correction);
			if(momentum_correction)
			{
				if(!momentum_correction->is_initialized())
					momentum_correction->initialize_previous_direction(variable_reference.vlen);
			}
		}

		for(index_t idx=0; idx<variable_reference.vlen; idx++)
		{
			float64_t negative_descend_direction=get_negative_descend_direction(
				variable_reference[idx], raw_negative_descend_direction[idx], idx, learning_rate);
			if(m_correction)
			{
				DescendPair pair=m_correction->get_corrected_descend_direction(
					negative_descend_direction, idx);
				variable_reference[idx]+=pair.descend_direction;
			}
			else
			{
				variable_reference[idx]-=negative_descend_direction;
			}
		}
	}

	/** Update a context object to store mutable variables
	 * used in descend update
	 *
	 * This method will be called by
	 * FirstOrderMinimizer::save_to_context()
	 *
	 * @param context, a context object
	 */
	virtual void update_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		if(m_correction)
			m_correction->update_context(context);
	}

	/** Load the given context object to restore mutable variables
	 *
	 * This method will be called by
	 * FirstOrderMinimizer::load_from_context(CMinimizerContext* context)
	 *
	 * @param context, a context object
	 */
	virtual void load_from_context(CMinimizerContext* context)
	{
		REQUIRE(context, "Context must set\n");
		if(m_correction)
			m_correction->load_from_context(context);
	}

	/** Set the type of descend correction
	 *
	 * @param correction the type of descend correction
	 */
	virtual void set_descend_correction(DescendCorrection* correction)
	{
		if(correction)
			m_correction=correction;
	}

	/** Do we enable descend correction?
	 *
	 * @return whether we enable descend correction
	 */
	virtual bool enables_descend_correction()
	{
		return m_correction!=NULL;
	}
protected:
	/** Get the negative descend direction given current variable and raw negative descend direction
	 *
	 * It will be called by update_variable()
	 *
	 * @param variable current variable
	 * @param raw_negative_descend_direction current raw negative descend direction
	 * @param idx the index of the variable
	 * @param learning_rate learning rate
	 * 
	 * @return negative descend direction
	 */
	virtual float64_t get_negative_descend_direction(float64_t variable,
		float64_t raw_negative_descend_direction, index_t idx, float64_t learning_rate)=0;

	/*  descend correction object */
	DescendCorrection* m_correction;

private:
	/*  Init */
	void init()
	{
		m_correction=NULL;
	}
};

}
#endif
