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

#ifndef SMDMINIMIZER_H
#define SMDMINIMIZER_H
#include <shogun/optimization/FirstOrderStochasticMinimizer.h>
#include <shogun/optimization/MappingFunction.h>
namespace shogun
{

/** @brief The class implements the stochastic mirror descend (SMD) minimizer.
 *
 */

class SMDMinimizer: public FirstOrderStochasticMinimizer
{
public:
	/** Default constructor */
	SMDMinimizer();

	/** Constructor
	 * @param fun stochastic cost function
	 */
	SMDMinimizer(FirstOrderStochasticCostFunction *fun);

	/** Destructor */
	virtual ~SMDMinimizer();

	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	virtual float64_t minimize();

	/** returns the name of the class
	 *
	 * @return name SMDMinimizer
	 */
	virtual const char* get_name() const { return "SMDMinimizer"; }

	/** Set projection function
	 * @param mapping_fun mapping/projection function
	 */
	virtual void set_mapping_function(MappingFunction* mapping_fun);

protected:
	/**  init the minimization process */
	virtual void init_minimization();

	/** mapping function */
	MappingFunction* m_mapping_fun;
private:
	  /* Init */
	void init();
};

}
#endif /* SMDMINIMIZER_H */
