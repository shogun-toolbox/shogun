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

#ifndef DESCENDCORRECTION_H
#define DESCENDCORRECTION_H
#include <shogun/base/SGObject.h>
namespace shogun
{
#ifndef DOXYGEN_SHOULD_SKIP_THIS
struct DescendPair
{
	DescendPair()
	{
		descend_direction=0.0;
		delta=0.0;
	}

	float64_t descend_direction;
	float64_t delta;
};
#endif
/** @brief This is a base class for descend based correction method.
 *
 * The interfact will be used in DescendUpdaterWithCorrection::update_variable()
 * An example of descend based correction is NesterovMomentumCorrection
 */
class DescendCorrection: public SGObject
{
public:
	/*  Constructor */
	DescendCorrection()
	{
		init();
	}

	/*  Destructor */
	~DescendCorrection() override {};

	/**  Set the weight used in descend correction
	 *
	 * param weight the weight
	 */
	virtual void set_correction_weight(float64_t weight);

	/** Get corrected descend direction
	 *
	 * @param negative_descend_direction the negative descend direction
	 * @param idx the index of the direction
	 * @return DescendPair (corrected descend direction and the change to correct descend direction)
	 */
	virtual DescendPair get_corrected_descend_direction(float64_t negative_descend_direction,
		index_t idx)=0;

protected:
	/**  weight of correction */
	float64_t m_weight;

private:
	/** Init */
	void init();
};

}
#endif
