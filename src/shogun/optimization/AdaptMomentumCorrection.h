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
#include <shogun/lib/SGVector.h>
#include <shogun/optimization/MomentumCorrection.h>
#include <shogun/mathematics/Math.h>

namespace shogun
{
/** @brief This implements the adaptive momentum correction method.
 *
 * A standard momentum correction performs update based on a momentum (eg, \f$\mu\f$), a previous descend direction (eg, \f$v\f$) and a
 * current descend direction (eg, \f$d\f$).
 *
 * The idea of adaptive momentum correction method is
 * If signs of the last two momentum corrections are different, the current descend direction is discounted
 * On the other hand, if the the signs are the same, the method raises the current descend direction 
 * Please see method get_corrected_descend_direction() for details
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

	/** returns the name of the class
	 *
	 * @return name AdaptMomentumCorrection
	 */
	virtual const char* get_name() const { return "AdaptMomentumCorrection"; }

	/** Set a standard momentum method 
	 * @param correction standard momentum method (eg, StandardMomentumCorrection)
	 *
	 */
	virtual void set_momentum_correction(MomentumCorrection* correction);

	/*  Destructor */
	virtual ~AdaptMomentumCorrection();

	/** Is the standard momentum method  initialized?
	 *
	 *  @return whether the standard method  is initialized
	 */
	virtual bool is_initialized();

	/** Set the weight (momentum) for the standard momentum method
	 *
	 * @param weight momentum
	 */
	virtual void set_correction_weight(float64_t weight);

	/**  Initialize m_previous_descend_direction
	 *
	 *  @return len the length of m_previous_descend_direction to be initialized
	 */
	/** Get corrected descend direction
	 *
	 * @param negative_descend_direction the negative descend direction
	 * @param idx the index of the direction
	 * @return DescendPair (corrected descend direction and the change to correct descend direction)
	*/
	virtual DescendPair get_corrected_descend_direction(float64_t negative_descend_direction,
		index_t idx);

	/**  Initialize m_previous_descend_direction
	 *
	 *  @return len the length of m_previous_descend_direction to be initialized
	 */
	virtual void initialize_previous_direction(index_t len);

	/** Set adaptive weights used in this method
	 *
	 * @param adapt_rate the rate is used to discount/raise the current descend direction (see get_corrected_descend_direction() )
	 * @param rate_min minimum of the rate
	 * @param rate_max maximum of the rate
	 */
	virtual void set_adapt_rate(float64_t adapt_rate, float64_t rate_min=0.0, float64_t rate_max=CMath::INFTY);

	/** Set the init rate used to discount/raise the current descend direction 
	 *
	 * @param init_descend_rate the init rate (default 1.0)
	 */
	virtual void set_init_descend_rate(float64_t init_descend_rate);

protected:
	/** element wise rate used to discount/raise the current descend direction  */
	SGVector<float64_t> m_descend_rate;
	/** the standard momentum method */
	MomentumCorrection* m_momentum_correction;
	/** the adapt rate */
	float64_t m_adapt_rate;
	/** the minimum of the adapt rate */
	float64_t m_rate_min;
	/** the maximum of the adapt rate */
	float64_t m_rate_max;
	/** the init rate */
	float64_t m_init_descend_rate;
private:
	/** Init */
	void init();

};

}
#endif
