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

#ifndef NESTEROVMOMENTUMCORRECTION_H
#define NESTEROVMOMENTUMCORRECTION_H
#include <shogun/lib/config.h>
#include <shogun/optimization/MomentumCorrection.h>
namespace shogun
{
/** @brief This implements the Nesterov's Accelerated Gradient (NAG) correction.
 *
 * Given a target variable, \f$w\f$, and a descend direction, \f$d_{ahead}\f$ wrt \f$w_{ahead}\f$,
 * the momentum method performs the following update:
 * \f{eqnarray*}{
 *   w_{ahead} &=&  w + \mu v \\
 *   v^{new}   &=& \mu v - d_{ahead} \\
 *   w^{new}   &=& w + v 
 * \f}
 * where \f$\mu\f$ is a momentum, \f$d_{ahead}\f$ is descend direction wrt \f$w_{ahead}\f$
 * (eg, \f$ d_{ahead}=\lambda g_{ahead}\f$, where \f$\lambda\f$ is learning rate, \f$g_{ahead}\f$ is gradient wrt \f$w_{ahead}\f$),
 * \f$v\f$ is a previous descend direction, and \f$v^{new}\f$ is a corrected descend direction.
 *
 * Note that the Nesterov momentum correction makes use of \f$d_{ahead}\f$  instead of \f$d\f$.
 *
 * In practice, we use the following implementation:
 * \f{eqnarray*}{
 *   v^{old} &=& v \\
 *   v^{new} &=& \mu  v^{old} - d \\
 *   w^{new} &=& w - \mu  v^{old} + (1 + \mu) v^{new}
 * \f}
 * where \f$d\f$ is descend direction wrt \f$w\f$
 *
 * The trick used in this implementation is we store \f$w_{ahead}\f$ and rename it as \f$w\f$.
 * Given a decay descend direction (eg, \f$d=\lambda g_{ahead}\f$, where \f$\lambda\f$ is a decay learning rate), \f$w_{ahead}\f$ is very close to \f$w\f$. 
 * When an optimal solution \f$w^{opt}\f$ is found, \f$w_{ahead}=w^{opt}\f$ since \f$d^{opt}=0\f$
 *
 * The get_corrected_descend_direction() method will do 
 * \f{eqnarray*}{
 *   v^{old} &=& v \\
 *   v^{new} &=& \mu  v^{old} - d
 * \f}
 * and return \f$ -\mu  v^{old} + (1 + \mu) v^{new}\f$
 *
 * A good introduction to the momentum update can be found at
 * http://cs231n.github.io/neural-networks-3/#sgd
 *
 * If you read the introduction at http://cs231n.github.io/neural-networks-3/#sgd ,
 * you may know that \f$v\f$ is also called velocity.
 */
class NesterovMomentumCorrection: public MomentumCorrection
{
public:
	/*  Constructor */
	NesterovMomentumCorrection()
		:MomentumCorrection()
	{
		init();
	}

	/*  Destructor */
	virtual ~NesterovMomentumCorrection() {}

	/** Get corrected descend direction
	 *
	 * @param negative_descend_direction the negative descend direction
	 * @param idx the index of the direction
	 * @return DescendPair (corrected descend direction and the change to correct descend direction)
	*/
	virtual DescendPair get_corrected_descend_direction(float64_t negative_descend_direction,
		index_t idx)
	{
		REQUIRE(idx>=0 && idx<m_previous_descend_direction.vlen,"The index (%d) is invalid\n", idx);
		float64_t tmp=m_weight*m_previous_descend_direction[idx];
		m_previous_descend_direction[idx]=tmp-negative_descend_direction;
		DescendPair pair;
		pair.delta=m_previous_descend_direction[idx];
		pair.descend_direction=(1.0+m_weight)*pair.delta-tmp;
		return pair;
	}
private:
	/*  Init */
	void init() { m_weight=0.9; }
};

}
#endif
