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

#ifndef SGDMINIMIZER_H
#define SGDMINIMIZER_H
#include <shogun/optimization/FirstOrderStochasticMinimizer.h>

namespace shogun
{

/** @brief The class implements the stochastic gradient descend (SGD) minimizer.
 *
 * A good introduction to SGD can be found at
 * http://cs231n.github.io/neural-networks-3/#sgd
 */

class SGDMinimizer: public FirstOrderStochasticMinimizer
{
public:
	/** Default constructor */
	SGDMinimizer();

	/** Constructor
	 * @param fun stochastic cost function
	 */
	SGDMinimizer(std::shared_ptr<FirstOrderStochasticCostFunction >fun);

	/** Destructor */
	~SGDMinimizer() override;

	/** returns the name of the class
	 *
	 * @return name SGDMinimizer
	 */
	const char* get_name() const override { return "SGDMinimizer"; }


	/** Do minimization and get the optimal value 
	 * 
	 * @return optimal value
	 */
	float64_t minimize() override;

protected:
	/*  init the minimization process */
	void init_minimization() override;

private:
	  /* Init */
	void init();
};

}
#endif /* SGDMINIMIZER_H */
