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

#ifndef WRAPPEDMINIMIZER_H
#define WRAPPEDMINIMIZER_H
#include <shogun/optimization/FirstOrderMinimizer.h>
namespace shogun
{

/** @brief The base class is used to wrap existing minimizers
 *
 * This class offers methods to work with existing minimizers
 *
 */
class CWrappedMinimizer: public CFirstOrderMinimizer
{
public:
	/** default constructor */
	CWrappedMinimizer();

	/** constructor
	 * @param fun cost function
	 */
	CWrappedMinimizer(CFirstOrderCostFunction *fun);

	/** destructor */
	virtual ~CWrappedMinimizer();

	/** return the name of a minimizer.
	 *
	 *  @return name 
	 */
	virtual const char* get_name() const=0;
protected:
	/** help function used to init existing minimizers
	 * Note that this function will convert our representation of variables
	 * to the classical vector representation
	 */
	virtual void minimization_init();

	/** help function used to convert our representation to the classical vector representation
	 * 
	 * @param input in our representation
	 *
	 * @return classical vector representation (this is a copy)
	 */
	static SGVector<float64_t> convert(CMap<TParameter*, SGVector<float64_t> >* input);

	/** help function used to copy input in the classical vector representation
	 * to output in our representation
	 * 
	 * @param input in the classical vector representation
	 * @param dim the length of input
	 * @param parameter_order the order of variables
	 * @param output in our representation
	 */

	static void copy_in_parameter_order(const float64_t* input, const int32_t dim,
		CMap<TParameter*, SGVector<float64_t> >* parameter_order,
		CMap<TParameter*, SGVector<float64_t> >* output);

	/** help function used to copy input in our representation to
	 * output in the classical vector representation
	 * 
	 * @param input in our representation
	 * @param parameter_order the order of variables
	 * @param output in the classical vector representation
	 * @param dim the length of output
	 */
	static void copy_in_parameter_order(CMap<TParameter*, SGVector<float64_t> >* input,
		CMap<TParameter*, SGVector<float64_t> >* parameter_order, float64_t* output, const int32_t dim);

	/* classical vector representation of variables*/
	SGVector<float64_t> m_variable_vec;
	/* our representation of variables*/
	CMap<TParameter*, SGVector<float64_t> >* m_variable;
	/* whether m_variable_vec is a reference of variables */
	bool m_is_in_place;
private:
	/*  init */
	void init();
};

}
#endif
