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

#ifndef CDESCENDUPDATER_H
#define CDESCENDUPDATER_H
#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/optimization/MinimizerContext.h>
namespace shogun
{
/** @brief This is a base class for descend update.
 *
 * the base class do the following job:
 * Given variable_reference and its negative_descend_direction
 * the update_variable function will do
 * variable_reference-=update_variable(negative_descend_direction)
 *
 * Note that an example of a negative_descend_direction is gradient  
 *
 */
class CDescendUpdater
{
public:
	/**update the variable_reference based on the negative_descend_direction
	 * 
	 * @param variable_reference a reference of the target variable
	 * @param negative_descend_direction the negative descend direction given the current variable
	 */
	virtual void update_variable(SGVector<float64_t> variable_reference,
		SGVector<float64_t> negative_descend_direction)=0;

	virtual void update_context(CMinimizerContext* context)=0;

	virtual void load_from_context(CMinimizerContext* context)=0;
};

}
#endif
